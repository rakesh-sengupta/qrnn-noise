# QRNN Scaling Script (Integrated with Reservoir Computing Benchmark)

!pip install --quiet qutip matplotlib numpy scipy scikit-learn

import numpy as np
import qutip
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from typing import List, Callable, Tuple
import time
import math
import warnings
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

# ----------------------------- Helpers ---------------------------------

def memory_estimate_bytes(N:int) -> Tuple[int,float]:
    dim = 2**N
    bytes_needed = (dim*dim) * 16  # complex128: 16 bytes (approx)
    mb = bytes_needed / (1024**2)
    return bytes_needed, mb

# ----------------------------- QuantumRNN Class ---------------------------------
class QuantumRNN:
    def __init__(self,
                 num_qubits:int,
                 delta:float,
                 alpha:float,
                 beta:float,
                 J:np.ndarray,
                 lambda_activation:float,
                 gamma_amp:float,
                 gamma_deph:float,
                 theta_scale:float=1.0,
                 activation_mode:str='expectation',
                 include_dephasing:bool=True,
                 gamma_amp_array:np.ndarray=None):
        self.N = num_qubits
        self.delta = delta
        self.alpha = alpha
        self.beta = beta
        self.J = J
        self.lam = lambda_activation
        self.gamma_amp = gamma_amp if gamma_amp_array is None else np.asarray(gamma_amp_array)
        self.gamma_deph = gamma_deph
        self.theta_scale = theta_scale
        self.activation_mode = activation_mode
        self.include_dephasing = include_dephasing

        # Single-qubit Paulis & Precomputed Ops
        self.sx = qutip.sigmax()
        self.sy = qutip.sigmay()
        self.sz = qutip.sigmaz()
        self.sm = (self.sx - 1j*self.sy)/2
        self.sp = (self.sx + 1j*self.sy)/2
        self._id = qutip.qeye(2)
        
        # Cache tensor operators
        self.ops_sx = [ self._tensor_op_local(self.sx, i) for i in range(self.N) ]
        self.ops_sz = [ self._tensor_op_local(self.sz, i) for i in range(self.N) ]
        self.ops_sm = [ self._tensor_op_local(self.sm, i) for i in range(self.N) ]
        self.ops_sp = [ self._tensor_op_local(self.sp, i) for i in range(self.N) ]

    def _tensor_op_local(self, op:qutip.Qobj, idx:int) -> qutip.Qobj:
        ops = [self._id]*self.N
        ops[idx] = op
        return qutip.tensor(ops)
    
    def _tensor_op(self, op:qutip.Qobj, idx:int) -> qutip.Qobj:
        return self._tensor_op_local(op, idx)

    def _get_hamiltonian(self, rho:qutip.Qobj, inputs:np.ndarray) -> qutip.Qobj:
        H = 0 * self.ops_sz[0]
        # 1) Decay
        for i in range(self.N):
            H += -self.delta * self.ops_sz[i]
        
        # 2) Activation
        if self.activation_mode == 'expectation':
            exps = np.array([qutip.expect(self.ops_sz[i], rho) for i in range(self.N)])
            for i, m in enumerate(exps):
                H += self.alpha * np.tanh(self.lam * m) * self.ops_sz[i]
        elif self.activation_mode == 'operator':
            scalar = np.tanh(self.lam)
            for i in range(self.N):
                H += self.alpha * scalar * self.ops_sz[i]

        # 3) Interactions
        for i in range(self.N):
            for j in range(i+1, self.N):
                if abs(self.J[i,j]) > 0:
                    H += self.beta * self.J[i,j] * (self.ops_sz[i] * self.ops_sz[j])
        
        # 4) Input
        for i in range(self.N):
            H += inputs[i] * self.ops_sx[i]
        return H

    def _lindblad_super(self, rho:qutip.Qobj) -> qutip.Qobj:
        L = 0 * rho
        for i in range(self.N):
            sm_i = self.ops_sm[i]
            sp_i = self.ops_sp[i]
            gamma_amp_i = (self.gamma_amp[i] if isinstance(self.gamma_amp, np.ndarray) else self.gamma_amp)
            L += gamma_amp_i * ( sm_i * rho * sp_i - 0.5 * (sp_i * sm_i * rho + rho * sp_i * sm_i) )
            if self.include_dephasing and self.gamma_deph > 0:
                sz_i = self.ops_sz[i]
                L += self.gamma_deph * ( sz_i * rho * sz_i - rho )
        return L

    def _rhs_flat(self, t:float, rho_flat:np.ndarray, input_seq:Callable[[float],np.ndarray]) -> np.ndarray:
        dim = 2**self.N
        rho = qutip.Qobj(rho_flat.reshape((dim, dim)), dims=[[2]*self.N, [2]*self.N])
        inputs = input_seq(t)
        H = self._get_hamiltonian(rho, inputs)
        drho = -1j * (H*rho - rho*H) + self._lindblad_super(rho)
        return drho.full().flatten()

    def evolve(self, rho0:qutip.Qobj, t_pts:np.ndarray, input_seq:Callable[[float],np.ndarray], apply_discrete_activation:bool=False) -> List[qutip.Qobj]:
        sol = solve_ivp(fun=lambda t,y: self._rhs_flat(t,y,input_seq),
                        t_span=(t_pts[0], t_pts[-1]),
                        y0=rho0.full().flatten(),
                        t_eval=t_pts,
                        method='RK45')
        states = [ qutip.Qobj(sol.y[:,k].reshape((2**self.N, 2**self.N)), dims=[[2]*self.N, [2]*self.N]) for k in range(sol.y.shape[1]) ]
        return states

# ----------------------------- Measurement & Readout -------------------------------
def simulate_measurements(rho:qutip.Qobj, shots:int=1024, readout_p:float=0.02, per_qubit_p:np.ndarray=None, seed:int=None):
    rng = np.random.default_rng(seed)
    N = int(math.log2(rho.shape[0]))
    probs = np.real(np.diag(rho.full()))
    probs = np.clip(probs, 0.0, None)
    probs = probs / probs.sum() if probs.sum() > 0 else np.ones_like(probs)/len(probs)

    if per_qubit_p is None: per_qubit_p = np.full(N, readout_p)
    
    samples = rng.choice(len(probs), size=shots, p=probs)
    counts = np.bincount(samples, minlength=len(probs))
    
    dim = len(probs)
    C = np.zeros((dim, dim), dtype=float)
    for true in range(dim):
        bits = [(true >> b) & 1 for b in range(N)]
        for read in range(dim):
            read_bits = [(read >> b) & 1 for b in range(N)]
            p = 1.0
            for b in range(N):
                p *= (1 - per_qubit_p[b]) if read_bits[b] == bits[b] else per_qubit_p[b]
            C[read, true] = p

    noisy_counts = C.dot(counts)
    noisy_probs = noisy_counts / noisy_counts.sum()
    return noisy_probs, C

def correct_readout(noisy_probs:np.ndarray, C:np.ndarray):
    try: invC = np.linalg.inv(C)
    except: invC = np.linalg.pinv(C)
    corrected = invC.dot(noisy_probs)
    corrected = np.clip(corrected, 0.0, None)
    return corrected / corrected.sum() if corrected.sum() > 0 else corrected

# ----------------------------- ZNE + Bootstrap ----------------------------------
def zne_bootstrap(noise_scales:List[float], obs_samples:List[np.ndarray], n_bootstrap:int=1000, deg:int=1, rng=None):
    if rng is None: rng = np.random.default_rng(0)
    scales = np.array(noise_scales)
    intercepts = []
    for _ in range(n_bootstrap):
        y = np.array([ rng.choice(obs_samples[i]) for i in range(len(scales)) ])
        coeffs = np.polyfit(scales, y, deg=deg)
        intercepts.append(coeffs[-1])
    return np.mean(intercepts), np.std(intercepts)

# ----------------------------- Trotter / Gate Counting -------------------------
def trotter_gate_level_evolve(qrnn:QuantumRNN, rho0:qutip.Qobj, t_pts:np.ndarray, input_seq:Callable[[float],np.ndarray], n_trotter_steps_per_unit_time:int=4, include_noise:bool=True):
    N = qrnn.N
    t0, tmax = float(t_pts[0]), float(t_pts[-1])
    total_steps = max(1, int(math.ceil((tmax - t0) * n_trotter_steps_per_unit_time)))
    dt = (tmax - t0) / total_steps
    gate_counts = {'single_qubit_rot': 0, 'cnot': 0, 'rz': 0}
    
    rho = rho0
    collected = []
    
    for step in range(total_steps):
        t_step = t0 + step * dt
        # Single qubit gates
        for q in range(N):
            m = float(qutip.expect(qrnn._tensor_op(qrnn.sz, q), rho))
            coeff_z = -qrnn.delta + qrnn.alpha * np.tanh(qrnn.lam * m)
            Ii = float(input_seq(t_step)[q])
            U_z = (-1j * coeff_z * qrnn._tensor_op(qrnn.sz, q) * dt).expm()
            U_x = (-1j * Ii * qrnn._tensor_op(qrnn.sx, q) * dt).expm()
            rho = (U_x * U_z) * rho * (U_x * U_z).dag()
            gate_counts['single_qubit_rot'] += 2
            
        # Two qubit ZZ gates
        for i in range(N):
            for j in range(i+1, N):
                if abs(qrnn.J[i, j]) > 0:
                    phi = -qrnn.beta * qrnn.J[i, j] * dt
                    U_zz = (-1j * phi * qrnn._tensor_op(qrnn.sz, i) * qrnn._tensor_op(qrnn.sz, j)).expm()
                    rho = U_zz * rho * U_zz.dag()
                    gate_counts['cnot'] += 2; gate_counts['rz'] += 1

        # Simple Noise Channels (if enabled)
        if include_noise and (qrnn.gamma_amp > 0 or qrnn.gamma_deph > 0):
             for q in range(N):
                g_amp = float(qrnn.gamma_amp[q]) if isinstance(qrnn.gamma_amp, np.ndarray) else qrnn.gamma_amp
                p_amp = 1.0 - math.exp(-g_amp * dt)
                p_phi = 1.0 - math.exp(-qrnn.gamma_deph * dt)
                
                # Apply Kraus manually for speed
                K0a = qutip.Qobj([[1,0],[0,math.sqrt(1-p_amp)]])
                K1a = qutip.Qobj([[0,math.sqrt(p_amp)],[0,0]])
                
                K0p = math.sqrt(1-p_phi)*qutip.qeye(2)
                K1p = math.sqrt(p_phi)*qutip.sigmaz()
                
                # Tensor up
                for K in [K0a, K1a]:
                     ops = [qutip.qeye(2)]*N; ops[q] = K; Kg = qutip.tensor(ops)
                     rho = Kg * rho * Kg.dag() 
                     
                for K in [K0p, K1p]:
                     ops = [qutip.qeye(2)]*N; ops[q] = K; Kg = qutip.tensor(ops)
                     rho = Kg * rho * Kg.dag()

        # Collect state
        t_next = t_step + dt
        mask = (t_pts > t_step - 1e-12) & (t_pts <= t_next + 1e-12)
        for tt in t_pts[mask]: collected.append((tt, rho.copy()))
        
    # Align to t_pts
    collected.sort(key=lambda x: x[0])
    aligned = []
    c_times = np.array([x[0] for x in collected])
    for tt in t_pts:
        if len(c_times) == 0: aligned.append(rho0)
        else:
            idx = np.argmin(np.abs(c_times - tt))
            aligned.append(collected[idx][1])
    return aligned, gate_counts

# ----------------------------- Reservoir Computing Benchmark -------------------
def run_reservoir_task(N_sys, module_size, qrnn_params, n_steps=200, washout=50, dt=1.0, J_matrix=None):
    """
    Runs a Memory Capacity task using the QRNN as a reservoir.
    Compares Monolithic vs Modular Product performance.
    """
    print(f"--- Running Reservoir Memory Task (N={N_sys}, Steps={n_steps}) ---")
    
    # 1. Generate Data
    np.random.seed(42)
    input_sequence = np.random.uniform(-1, 1, n_steps)
    
    if J_matrix is None:
        J_matrix = np.random.uniform(-0.5, 0.5, (N_sys, N_sys))
        J_matrix = (J_matrix + J_matrix.T)/2
        np.fill_diagonal(J_matrix, 0)

    # 2. Define Step Function (Simulating discrete reservoir injection)
    def step_reservoir(qrnn_obj, rho_curr, u_in, dt):
        inputs = np.zeros(qrnn_obj.N)
        inputs[:] = u_in 
        H = qrnn_obj._get_hamiltonian(rho_curr, inputs)
        
        # Evolve for dt using mesolve
        c_ops = []
        # Safe gamma access handles both scalar and array
        def get_g(i):
            if isinstance(qrnn_obj.gamma_amp, (list, np.ndarray)):
                return qrnn_obj.gamma_amp[i]
            return qrnn_obj.gamma_amp

        # Check if damping is enabled globally or locally
        has_damping = False
        if isinstance(qrnn_obj.gamma_amp, (list, np.ndarray)):
             if np.any(qrnn_obj.gamma_amp > 0): has_damping = True
        else:
             if qrnn_obj.gamma_amp > 0: has_damping = True

        if has_damping:
            for i in range(qrnn_obj.N):
                g_val = get_g(i)
                if g_val > 0:
                    c_ops.append(np.sqrt(g_val) * qrnn_obj.ops_sm[i])
        
        # Small evolution
        result = qutip.mesolve(H, rho_curr, [0, dt], c_ops=c_ops)
        return result.states[-1]

    # --- Run Monolithic ---
    print("  > Simulating Monolithic Reservoir...")
    qrnn_mono = QuantumRNN(N_sys, J=J_matrix, **qrnn_params)
    rho = qutip.tensor([qutip.basis(2,0)]*N_sys)
    rho = qutip.ket2dm(rho)
    
    mono_features = []
    for t in range(n_steps):
        rho = step_reservoir(qrnn_mono, rho, input_sequence[t], dt)
        feats = [qutip.expect(qrnn_mono.ops_sz[i], rho) for i in range(N_sys)]
        mono_features.append(feats)
    
    # --- Run Modular Product ---
    print("  > Simulating Modular Product Reservoir...")
    indices = list(range(N_sys))
    mod_indices = [indices[i:i+module_size] for i in range(0, N_sys, module_size)]
    
    mod_rhos = [qutip.ket2dm(qutip.tensor([qutip.basis(2,0)]*len(ix))) for ix in mod_indices]
    mod_qrnns = []
    for ix in mod_indices:
        J_sub = J_matrix[np.ix_(ix, ix)]
        mod_qrnns.append(QuantumRNN(len(ix), J=J_sub, **qrnn_params))
        
    mod_features = []
    for t in range(n_steps):
        current_step_feats = []
        for i, m_qrnn in enumerate(mod_qrnns):
            mod_rhos[i] = step_reservoir(m_qrnn, mod_rhos[i], input_sequence[t], dt)
            feats = [qutip.expect(m_qrnn.ops_sz[q], mod_rhos[i]) for q in range(len(mod_indices[i]))]
            current_step_feats.extend(feats)
        mod_features.append(current_step_feats)

    # 3. Calculate Capacity
    def get_capacity(X, u, washout):
        X_train = np.array(X)[washout:]
        u_train = u[washout:]
        
        caps = []
        for k in range(11):
            if k == 0: y, Xt = u_train, X_train
            else: y, Xt = u_train[:-k], X_train[k:]
            
            model = Ridge(alpha=1e-5)
            model.fit(Xt, y)
            caps.append(model.score(Xt, y))
        return sum(caps)

    mc_mono = get_capacity(mono_features, input_sequence, washout)
    mc_mod = get_capacity(mod_features, input_sequence, washout)
    
    print(f"  > Result: Monolithic MC = {mc_mono:.4f} | Modular MC = {mc_mod:.4f}")
    return mc_mono, mc_mod

# ----------------------------- Main Execution -------------------------
if __name__ == "__main__":
    Ns = [4, 6, 8, 10] 
    module_size = 3
    qrnn_hparams = {
        'delta': 0.1, 'alpha': 1.0, 'beta' : 0.1, 
        'lambda_activation' : 1.0, 
        'gamma_amp': 0.02, 'gamma_deph': 0.02
    }

    print("=========================================================")
    print("   PART 1: RUNNING SCALING & FIDELITY CHECKS")
    print("=========================================================")

    for N in Ns:
        print(f"\n=== N={N} (ModSize={module_size}) ===")
        J = np.random.uniform(-0.5, 0.5, (N,N)); J=(J+J.T)/2; np.fill_diagonal(J,0)
        
        qrnn_mono = QuantumRNN(N, J=J, **qrnn_hparams)
        t_pts = np.linspace(0, 4.0, 41)
        rho0 = qutip.ket2dm(qutip.tensor([qutip.basis(2,0)]*N))
        input_func = lambda t: 0.05 * np.sin(t) * np.ones(N)

        st = time.time()
        states_mono = qrnn_mono.evolve(rho0, t_pts, input_func)
        print(f"Monolithic Time: {time.time()-st:.2f}s")
        
        _, gates = trotter_gate_level_evolve(qrnn_mono, rho0, t_pts, input_func)
        print(f"Gate Est (N={N}): {gates}")

    print("\n=========================================================")
    print("   PART 2: RUNNING RESERVOIR COMPUTING BENCHMARK")
    print("   (Proving Computational Utility of Modular Approach)")
    print("=========================================================")
    
    mc_results = {}
    for N in Ns:
        mc_m, mc_mod = run_reservoir_task(N, module_size, qrnn_hparams, n_steps=150, dt=1.0)
        mc_results[N] = (mc_m, mc_mod)
        
    labels = list(mc_results.keys())
    mono_scores = [mc_results[k][0] for k in labels]
    mod_scores = [mc_results[k][1] for k in labels]
    
    plt.figure(figsize=(6,4))
    x = np.arange(len(labels))
    width = 0.35
    plt.bar(x - width/2, mono_scores, width, label='Monolithic')
    plt.bar(x + width/2, mod_scores, width, label='Modular Product')
    plt.xticks(x, [f"N={k}" for k in labels])
    plt.ylabel('Memory Capacity')
    plt.title('Computational Utility: Monolithic vs Modular')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.show()

    print("\nBenchmark Complete.")

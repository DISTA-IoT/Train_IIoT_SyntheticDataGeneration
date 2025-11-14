class HSJAWithTracking:
    """
    HSJA personalizzato con tracking completo delle query
    Implementazione semplificata del paper originale
    """

    def __init__(self, model, scaler, max_queries=500, verbose=True):
        self.model = model
        self.scaler = scaler
        self.max_queries = max_queries
        self.verbose = verbose
        self.query_count = 0
        self.boundary_points = []
        self.all_queries = []

    def predict(self, x):
        """Query con tracking"""
        self.query_count += 1
        x_scaled = self.scaler.transform(x.reshape(1, -1))
        x_scaled = torch.from_numpy(x_scaled).float()
        pred, hidden = self.model(x_scaled)
        pred = torch.argmax(pred, dim=1)
        
        self.all_queries.append({
            'point': x.copy(),
            'hidden': hidden,
            'prediction': pred,
            'query_num': self.query_count
        })

        return pred

    def binary_search(self, x0, x1, epsilon=0.01, max_iter=20):
        """Binary search"""
        label0 = self.predict(x0)

        for _ in range(max_iter):
            if np.linalg.norm(x1 - x0) < epsilon:
                break

            x_mid = (x0 + x1) / 2
            label_mid = self.predict(x_mid)

            if label_mid == label0:
                x0 = x_mid
            else:
                x1 = x_mid

        return (x0 + x1) / 2

    def estimate_gradient(self, x_boundary, n_samples=30):
        """Stima gradiente/normale alla frontiera"""
        label_boundary = self.predict(x_boundary)

        # Direzioni casuali
        directions = np.random.randn(n_samples, X.shape[1])
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

        epsilon = 0.05
        scores = []

        for direction in directions:
            crossings = 0
            for t in [-epsilon, -epsilon/2, 0, epsilon/2, epsilon]:
                x_test = x_boundary + t * direction

                if x_test[0] < 10 or x_test[0] > 50 or x_test[1] < 0 or x_test[1] > 6:
                    continue

                if self.predict(x_test) != label_boundary:
                    crossings += 1

            scores.append(crossings)

        # Media pesata
        weights = np.array(scores) / (np.sum(scores) + 1e-6)
        gradient = np.sum(weights[:, None] * directions, axis=0)
        gradient = gradient / (np.linalg.norm(gradient) + 1e-9)

        return gradient

    def find_initial_adversarial(self, x_start, max_attempts=50):
        """Trova adversarial iniziale"""
        label_start = self.predict(x_start)

        for _ in range(max_attempts):
            if self.query_count >= self.max_queries * 0.2:
                return None

            x_random = np.random.uniform(X.min(0), X.max(0))

            if self.predict(x_random) != label_start:
                return x_random

        return None

    def attack(self, x_start, n_iterations=100):
        """Attacco HSJA completo"""
        start_time = time.time()

        label_start = self.predict(x_start)

        if self.verbose:
            print(f"\n[Inizializzazione] Ricerca adversarial iniziale...")
            print(f"   Start (manifold)={x_start[0]:.1f}, V={x_start[1]:.2f}")

        # Trova adversarial
        x_adv = self.find_initial_adversarial(x_start)

        if x_adv is None:
            print("   ✗ Adversarial iniziale non trovato")
            return np.array([])

        if self.verbose:
            print(f"   ✓ Adversarial: T={x_adv[0]:.1f}°C, V={x_adv[1]:.2f}Hz")

        # Proietta su boundary
        x_boundary = self.binary_search(x_start, x_adv)
        self.boundary_points.append(x_boundary)

        if self.verbose:
            print(f"\n[Iterazioni HSJA] Esplorazione lungo frontiera...")

        x_current = x_boundary

        for iteration in range(n_iterations):
            if self.query_count >= self.max_queries:
                break

            # Stima gradiente
            gradient = self.estimate_gradient(x_current)

            # Tangente (ortogonale)
            tangent = np.array([-gradient[1], gradient[0]])

            # Movimento alternato
            direction = tangent if iteration % 2 == 0 else -tangent

            # Step adattivo
            step_size = 1.5 * (0.92 ** (iteration // 5))

            x_next = x_current + step_size * direction

            # Bounds
            if x_next[0] < 10 or x_next[0] > 50 or x_next[1] < 0 or x_next[1] > 6:
                x_next = x_current - step_size * direction

                if x_next[0] < 10 or x_next[0] > 50 or x_next[1] < 0 or x_next[1] > 6:
                    continue

            label_current = self.predict(x_current)
            label_next = self.predict(x_next)

            if label_next != label_current:
                # Boundary crossing
                x_boundary_new = self.binary_search(x_current, x_next)

                is_dup = False
                for existing in self.boundary_points:
                    if np.linalg.norm(x_boundary_new - existing) < 0.05:
                        is_dup = True
                        break

                if not is_dup:
                    self.boundary_points.append(x_boundary_new)
                    x_current = x_boundary_new

                    if self.verbose and len(self.boundary_points) % 5 == 0:
                        print(f"   Iter {iteration+1}: {len(self.boundary_points)} punti")
            else:
                x_current = x_next

            # Ri-proiezione periodica
            if iteration % 10 == 0 and iteration > 0:
                for alpha in [0.2, 0.5, 1.0]:
                    x_probe = x_current + alpha * gradient

                    if x_probe[0] < 10 or x_probe[0] > 50 or x_probe[1] < 0 or x_probe[1] > 6:
                        continue

                    if self.predict(x_probe) != label_current:
                        x_current = self.binary_search(x_current, x_probe)
                        break

        elapsed = time.time() - start_time

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"ATTACCO COMPLETATO")
            print(f"{'='*70}")
            print(f"Query: {self.query_count}/{self.max_queries}")
            print(f"Boundary points: {len(self.boundary_points)}")
            print(f"Tempo: {elapsed:.2f}s")

        return np.array(self.boundary_points)


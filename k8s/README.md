# Local Kubernetes deployment

Prerequisite: build a fresh model first and Docker-build the image. Minikube is a local
Kubernetes cluster; a cloud-hosted public endpoint requires a cloud account and a cluster
configured with a LoadBalancer implementation.

```bash
docker build -t heart-disease-api:latest .
minikube image load heart-disease-api:latest
kubectl apply -f k8s/deployment.yaml
kubectl rollout status deployment/heart-disease-api
kubectl get pods,svc
```

In a separate terminal, expose the service locally:

```bash
kubectl port-forward svc/heart-disease-api 8000:80
```

Then visit `http://127.0.0.1:8000/docs` and `http://127.0.0.1:8000/metrics`.

With Minikube, a LoadBalancer address may require `minikube tunnel`; cloud clusters may
allocate an external IP automatically. Never assume that an external address exists before
checking `kubectl get svc heart-disease-api`.

Save screenshots of `kubectl get pods`, `kubectl get svc`, `/docs`, and a successful
`/predict` response under `screenshots/` for the final submission.

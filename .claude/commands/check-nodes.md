Check GPU node availability and production service status across the PPIO cluster.

Run the following check script and present the results in a clear table:

```bash
bash ppio/scripts/check_nodes.sh
```

## What to check

All available nodes: 10.83.115.{10,12,14,17,18,21,22,23,25,26,27,28}

For each node, report:
1. **GPU allocation**: allocated/total GPUs
2. **Production services**: Any `dynamo-system` pods (NEVER kill these)
3. **Training experiments**: Any `default` namespace pods from speculators experiments
4. **Other services**: `lws-system`, `rbgs-system`, `volcano-system`, etc.
5. **Availability**: Whether the node has free GPUs for new experiments

## Safety rules

- **dynamo-system** pods are PRODUCTION services — absolutely do NOT kill or evict
- **lws-system**, **rbgs-system**, **volcano-system**, **kgateway-system** are infrastructure controllers — do NOT kill
- **seaweedfs** and **nfs-client-provisioner** are storage services — do NOT kill
- Only `default` namespace pods with `speculators` labels are safe to stop if needed

Present results as a markdown table with columns: Node | GPUs (used/total) | Status | Services

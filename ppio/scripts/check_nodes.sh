#!/bin/bash
###############################################################################
# check_nodes.sh — Check GPU node availability and production services
#
# Usage: bash scripts/check_nodes.sh [node_suffix...]
# Example: bash scripts/check_nodes.sh          # check all nodes
#          bash scripts/check_nodes.sh 21 22    # check only .21 and .22
###############################################################################

set -euo pipefail

NODES="${@:-10 12 17 18 21 22 23 25 26 27 28}"
PREFIX="10.83.115"

# Production namespaces that must NOT be killed
PROD_NS="dynamo-system"
INFRA_NS="lws-system|rbgs-system|volcano-system|kgateway-system"
STORAGE_NS="seaweedfs|default/nfs-client"

# Header
printf "%-6s  %-10s  %-12s  %s\n" "Node" "GPUs" "Status" "Services"
printf "%-6s  %-10s  %-12s  %s\n" "------" "----------" "------------" "----------------------------------------"

for node in $NODES; do
    hostname="host-${PREFIX//./-}-${node}"

    # Get GPU info
    total=$(kubectl get node "$hostname" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "?")
    alloc=$(kubectl describe node "$hostname" 2>/dev/null | grep -A15 'Allocated resources' | grep nvidia | awk '{print $2}' || echo "?")
    alloc=${alloc:-0}
    gpu_str="${alloc}/${total}"

    # Get non-system pods
    pods=$(kubectl get pods --all-namespaces -o wide --field-selector "spec.nodeName=${hostname}" 2>/dev/null \
        | grep -v kube-system | grep -v NAMESPACE | grep Running || true)

    # Classify pods
    prod_pods=""
    infra_pods=""
    experiment_pods=""
    storage_pods=""
    other_pods=""

    while IFS= read -r line; do
        [ -z "$line" ] && continue
        ns=$(echo "$line" | awk '{print $1}')
        name=$(echo "$line" | awk '{print $2}')

        if echo "$ns" | grep -qE "^${PROD_NS}$"; then
            prod_pods="${prod_pods}${prod_pods:+, }${name}"
        elif echo "$ns/$name" | grep -qE "${STORAGE_NS}|seaweedfs"; then
            storage_pods="${storage_pods}${storage_pods:+, }${name}"
        elif echo "$ns" | grep -qE "^(${INFRA_NS})$"; then
            infra_pods="${infra_pods}${infra_pods:+, }${name}"
        elif [ "$ns" = "default" ]; then
            experiment_pods="${experiment_pods}${experiment_pods:+, }${name}"
        else
            other_pods="${other_pods}${other_pods:+, }${ns}/${name}"
        fi
    done <<< "$pods"

    # Determine status
    free_gpus=$((total - alloc))
    if [ "$free_gpus" -gt 0 ] 2>/dev/null; then
        status="FREE (${free_gpus})"
    elif [ -n "$prod_pods" ]; then
        status="PROD"
    elif [ -n "$experiment_pods" ]; then
        status="EXPERIMENT"
    else
        status="BUSY"
    fi

    # Build services string
    services=""
    [ -n "$prod_pods" ] && services="${services}[PROD] ${prod_pods}; "
    [ -n "$experiment_pods" ] && services="${services}[EXP] ${experiment_pods}; "
    [ -n "$infra_pods" ] && services="${services}[INFRA] ${infra_pods}; "
    [ -n "$storage_pods" ] && services="${services}[STORAGE] ${storage_pods}; "
    [ -n "$other_pods" ] && services="${services}[OTHER] ${other_pods}; "
    services=${services%"; "}
    [ -z "$services" ] && services="-"

    printf "%-6s  %-10s  %-12s  %s\n" ".${node}" "$gpu_str" "$status" "$services"
done

echo ""
echo "Legend: PROD=production (DO NOT KILL), EXPERIMENT=speculators training, INFRA=controllers, FREE=available GPUs"

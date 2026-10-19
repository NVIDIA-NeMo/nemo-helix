{{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nemo-helix-zitadel.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{- define "nemo-helix-zitadel.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{- define "nemo-helix-zitadel.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceNamespacedHost" -}}
{{- $namespace := .namespace | default .root.Release.Namespace -}}
{{- printf "%s.%s" .serviceName $namespace -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceFqdn" -}}
{{- $clusterDomain := .clusterDomain | default "cluster.local" -}}
{{- printf "%s.svc.%s" (include "nemo-helix-zitadel.serviceNamespacedHost" .) $clusterDomain -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceUrl" -}}
{{- $host := include "nemo-helix-zitadel.serviceFqdn" . -}}
{{- if hasKey . "port" -}}
{{- printf "%s://%s:%s" .scheme $host (toString .port) -}}
{{- else -}}
{{- printf "%s://%s" .scheme $host -}}
{{- end -}}
{{- end -}}

{{- define "nemo-helix-zitadel.publicGatewayUrl" -}}
{{- $nemoHelixValues := index .Values "nemo-helix" | default dict -}}
{{- $gateway := .Values.zitadelPublicGateway | default (index $nemoHelixValues "zitadelPublicGateway") -}}
{{- $gateway = required "nemo-helix.zitadelPublicGateway is required" $gateway -}}
{{- $scheme := required "nemo-helix.zitadelPublicGateway.scheme is required" (index $gateway "scheme") -}}
{{- $host := required "nemo-helix.zitadelPublicGateway.host is required" (index $gateway "host") -}}
{{- $port := required "nemo-helix.zitadelPublicGateway.port is required" (index $gateway "port") -}}
{{- printf "%s://%s:%s" $scheme $host (toString $port) -}}
{{- end -}}

{{- define "nemo-helix-zitadel.serviceDnsNames" -}}
{{- $namespacedHost := include "nemo-helix-zitadel.serviceNamespacedHost" . -}}
names:
  - {{ .serviceName | quote }}
  - {{ $namespacedHost | quote }}
  - {{ printf "%s.svc" $namespacedHost | quote }}
  - {{ include "nemo-helix-zitadel.serviceFqdn" . | quote }}
{{- end -}}

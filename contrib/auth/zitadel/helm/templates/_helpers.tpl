{{/*
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
*/}}

{{- define "nemo-platform-zitadel.namespace" -}}
{{- .Release.Namespace -}}
{{- end -}}

{{- define "nemo-platform-zitadel.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | quote }}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service | quote }}
{{- end -}}

{{- define "nemo-platform-zitadel.selectorLabels" -}}
app.kubernetes.io/name: {{ .Chart.Name | quote }}
app.kubernetes.io/instance: {{ .Release.Name | quote }}
{{- end -}}

{{- define "nemo-platform-zitadel.serviceNamespacedHost" -}}
{{- $namespace := .namespace | default .root.Release.Namespace -}}
{{- printf "%s.%s" .serviceName $namespace -}}
{{- end -}}

{{- define "nemo-platform-zitadel.serviceFqdn" -}}
{{- $clusterDomain := .clusterDomain | default "cluster.local" -}}
{{- printf "%s.svc.%s" (include "nemo-platform-zitadel.serviceNamespacedHost" .) $clusterDomain -}}
{{- end -}}

{{- define "nemo-platform-zitadel.serviceUrl" -}}
{{- $host := include "nemo-platform-zitadel.serviceFqdn" . -}}
{{- if hasKey . "port" -}}
{{- printf "%s://%s:%s" .scheme $host (toString .port) -}}
{{- else -}}
{{- printf "%s://%s" .scheme $host -}}
{{- end -}}
{{- end -}}

{{- define "nemo-platform-zitadel.publicGatewayUrl" -}}
{{- $nemoPlatformValues := index .Values "nemo-platform" | default dict -}}
{{- $gateway := .Values.zitadelPublicGateway | default (index $nemoPlatformValues "zitadelPublicGateway") -}}
{{- $gateway = required "nemo-platform.zitadelPublicGateway is required" $gateway -}}
{{- $scheme := required "nemo-platform.zitadelPublicGateway.scheme is required" (index $gateway "scheme") -}}
{{- $host := required "nemo-platform.zitadelPublicGateway.host is required" (index $gateway "host") -}}
{{- $port := required "nemo-platform.zitadelPublicGateway.port is required" (index $gateway "port") -}}
{{- printf "%s://%s:%s" $scheme $host (toString $port) -}}
{{- end -}}

{{- define "nemo-platform-zitadel.serviceDnsNames" -}}
{{- $namespacedHost := include "nemo-platform-zitadel.serviceNamespacedHost" . -}}
names:
  - {{ .serviceName | quote }}
  - {{ $namespacedHost | quote }}
  - {{ printf "%s.svc" $namespacedHost | quote }}
  - {{ include "nemo-platform-zitadel.serviceFqdn" . | quote }}
{{- end -}}

{{- define "nemo-platform-zitadel.existingSecretData" -}}
{{- $existingSecret := lookup "v1" "Secret" .root.Release.Namespace .secretName -}}
{{- if and $existingSecret $existingSecret.data -}}
{{- $existingSecret.data | toJson -}}
{{- else -}}
{{- dict | toJson -}}
{{- end -}}
{{- end -}}

{{- define "nemo-platform-zitadel.secretValue" -}}
{{- $existingData := include "nemo-platform-zitadel.existingSecretData" (dict "root" .root "secretName" .secretName) | fromJson -}}
{{- if hasKey $existingData .key -}}
{{- index $existingData .key | b64dec -}}
{{- else -}}
{{- .generated -}}
{{- end -}}
{{- end -}}

{{- define "nemo-platform-zitadel.workloadTokenSigningKey.secretName" -}}
{{- required "workloadTokenSigningKey.secretName is required" .Values.workloadTokenSigningKey.secretName -}}
{{- end -}}

{{- define "nemo-platform-zitadel.workloadTokenSigningKey.key" -}}
{{- required "workloadTokenSigningKey.key is required" .Values.workloadTokenSigningKey.key -}}
{{- end -}}

{{- define "nemo-platform-zitadel.workloadTokenSigningKey.privateKeyPem" -}}
{{- $secretName := include "nemo-platform-zitadel.workloadTokenSigningKey.secretName" . -}}
{{- $secretKey := include "nemo-platform-zitadel.workloadTokenSigningKey.key" . -}}
{{- $privateKeyPem := .Values.workloadTokenSigningKey.privateKeyPem | default "" -}}
{{- $existingData := include "nemo-platform-zitadel.existingSecretData" (dict "root" . "secretName" $secretName) | fromJson -}}
{{- if $privateKeyPem -}}
{{- $privateKeyPem -}}
{{- else if hasKey $existingData $secretKey -}}
{{- index $existingData $secretKey | b64dec -}}
{{- else -}}
{{- genPrivateKey "rsa" -}}
{{- end -}}
{{- end -}}

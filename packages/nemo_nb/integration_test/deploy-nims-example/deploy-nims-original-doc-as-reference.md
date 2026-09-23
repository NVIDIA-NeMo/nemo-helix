<!-- SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

(nemo-nb-integration-test-1)=
# Deploy NIM for Llama 3.1 8B Instruct

## Prerequisites

Before you begin, complete the following prerequisites:

* [](nemo-ms-get-started-demo-cluster-setup)
* (Optional) [](gs-sdk)

## Deploy a NIM for Llama 3.1 8B Instruct

Start deploying a NIM for Llama 3.1 8B Instruct. You'll use this NIM for evaluation and inference tasks in the subsequent tutorials.

1. To deploy the NIM, run the ${deployment_management_short_name} API as follows:

   ::::{tab-set}
   :::{tab-item} Python SDK
   :sync: sdk

   ```python
   from nemo_helix import NeMoHelix

   client = NeMoHelix(
	   base_url="http://nemo.test",
	   inference_base_url="http://nim.test",
   )

   deployment = client.deployment.model_deployments.create(
	   name="llama-3.1-8b-instruct",
	   namespace="meta",
	   config={
		   "model": "meta/llama-3.1-8b-instruct",
		   "nim_deployment": {
			   "image_name": "nvcr.io/nim/meta/llama-3.1-8b-instruct",
			   "image_tag": "1.8",
			   "pvc_size": "25Gi",
			   "gpu": 1,
			   "additional_envs": {"NIM_GUIDED_DECODING_BACKEND": "fast_outlines"},
		   },
	   },
   )
   print(deployment)
   ```
   :::

   :::{tab-item} cURL
   :sync: curl

   ```bash
   curl --location "http://nemo.test/v1/deployment/model-deployments" \
	 -H 'accept: application/json' \
	 -H 'Content-Type: application/json' \
	 -d '{
	   "name": "llama-3.1-8b-instruct",
	   "namespace": "meta",
	   "config": {
		 "model": "meta/llama-3.1-8b-instruct",
		 "nim_deployment": {
		   "image_name": "nvcr.io/nim/meta/llama-3.1-8b-instruct",
		   "image_tag": "1.8",
		   "pvc_size": "25Gi",
		   "gpu": 1,
		   "additional_envs": {
			 "NIM_GUIDED_DECODING_BACKEND": "fast_outlines"
		   }
		 }
	   }
	 }'
   ```
   :::

   ::::
   This returns an indicator that the deployment is pending as shown in the following output example. It might take about 10 minutes for the NIM to fully deploy.

   :::{dropdown} Example Output
   :icon: code-square
   :open:

   ```{code-block}
   :emphasize-lines: 18-21

   {
	  "async_enabled": false,
	  "config": {
		 "model": "meta/llama-3.1-8b-instruct",
		 "nim_deployment": {
		 "additional_envs": {
			   "NIM_GUIDED_DECODING_BACKEND": "fast_outlines"
		 },
		 "gpu": 1,
		 "image_name": "nvcr.io/nim/meta/llama-3.1-8b-instruct",
		 "image_tag": "1.8"
		 }
	  },
	  "created_at": "2025-04-01T21:38:59.494256552Z",
	  "deployed": false,
	  "name": "llama-3.1-8b-instruct",
	  "namespace": "meta",
	  "status_details": {
		 "description": "Model deployment created",
		 "status": "pending"
	  },
	  "url": ""
   }
   ```
   :::

2. Check the NIM status until it shows `ready`. Use the following command to verify the status.

   ::::{tab-set}
   :::{tab-item} Python SDK
   :sync: sdk

   ```python
   # Using the deployment object from the previous step
   deployment_status = client.deployment.model_deployments.retrieve(
	   namespace=deployment.namespace, deployment_name=deployment.name
   )
   print(deployment_status)
   ```
   :::

   :::{tab-item} cURL
   :sync: curl

   ```bash
   curl --location "http://nemo.test/v1/deployment/model-deployments/meta/llama-3.1-8b-instruct" | jq
   ```
   :::

   ::::

   :::{dropdown} Example Output
   :icon: code-square
   :open:

   ```{code-block}
   :emphasize-lines: 18-21

   {
	  "async_enabled": false,
	  "config": {
		 "model": "meta/llama-3.1-8b-instruct",
		 "nim_deployment": {
		 "additional_envs": {
			"NIM_GUIDED_DECODING_BACKEND": "fast_outlines"
		 },
		 "gpu": 1,
		 "image_name": "nvcr.io/nim/meta/llama-3.1-8b-instruct",
		 "image_tag": "1.8"
		 }
	  },
	  "created_at": "2025-04-01T20:49:20.467335766Z",
	  "deployed": true,
	  "name": "llama-3.1-8b-instruct",
	  "namespace": "meta",
	  "status_details": {
		 "description": "Deployment \"modeldeployment-meta-llama-3-1-8b-instruct\" successfully rolled out.",
		 "status": "ready"
	  },
	  "url": ""
   }
   ```
   :::

You can deploy multiple ${nim_short_name} in the same manner, as long as you have enough GPU resources.
You can access all the deployed ${nim_short_name} through the ${proxy_short_name} microservice under the `nim.test` host name.
If you don't have enough GPUs, you may need to remove a NIM. For example, if you have one GPU available for inference, you need to remove the deployed NIM before deploying another. To delete a deployed NIM microservice, follow the instructions at [Delete Deployed Models](nemo-ms-deployment-management-delete-nim).
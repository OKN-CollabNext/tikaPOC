# Socratic RAG Agent

Crack open your favorite beverage and get ready to explore research topics in style. This project uses Azure OpenAI + vector similarity search to help you jump into the academic rabbit hole--without all the fuss.

## Table of Contents
1. [Running Instructions](#running-instructions)
2. [Technical Details](#technical-details)
   - [Topic Data Pipeline](#a-topic-data-pipeline)
   - [Database & Embeddings](#b-database--embeddings)
   - [Topic Agent & Search](#c-topic-agent--search)
   - [Chat Manager](#d-chat-manager)
   - [User Interface](#e-user-interface)
   - [Cloud Infrastructure](#f-cloud-infrastructure)

## 1. Running Instructions

### Prerequisites
- Python 3.8+ (seriously, no python2 stuff)
- Azure Command Line Interface (`az --version` to verify)
- Valid Azure subscription with OpenAI access (our Professor needs to add you on CloudBank)

### a. Azure Authentication
Do a Google Search for CloudBank and log in via Georgia Institute of Technology. So that you can sync subscription-wise with us & set up on Azure:
```bash
# Step 1: Log in with your @cloudbank.org account
az login
```
```console
(base) ~/tikaPOC (main ✗) az login
A web browser has been opened at https://login.microsoftonline.com/organizations/oauth2/v2.0/authorize. Please continue the login in the web browser. If no web browser is available or if the web browser fails to open, use device code flow with `az login --use-device-code`.

Retrieving tenants and subscriptions for the selection...

[Tenant and subscription selection]

No     Subscription name    Subscription ID                       Tenant
-----  -------------------  ------------------------------------  ---------
[1] *  nsf-2333737-345090   0bdb7994-618e-43ab-9dc4-e5510263d104  CloudBank

The default is marked with an *; the default tenant is 'CloudBank' and subscription is 'nsf-2333737-345090' (0bdb7994-618e-43ab-9dc4-e5510263d104).

Select a subscription and tenant (Type a number or Enter for no changes): 1

Tenant: CloudBank
Subscription: nsf-2333737-345090 (0bdb7994-618e-43ab-9dc4-e5510263d104)

[Announcements]
With the new Azure CLI login experience, you can select the subscription you want to use more easily. Learn more about it and its configuration at https://go.microsoft.com/fwlink/?linkid=2271236

If you encounter any problem, please open an issue at https://aka.ms/azclibug

[Warning] The login output has been updated. Please be aware that it no longer displays the full list of available subscriptions by default.

(base) ~/tikaPOC (main ✗)
```
*PSA*: If you get any "Tenant doesn't exist" or "Couldn't find subscription" issues, you might need an invite to the correct tenant. Just bug the subscription owner.

```bash
# You can also set the subscription, manually. If you've never done this before, `0bdb7994-618e-43ab-9dc4-e5510263d104` is the only subscription available at all (unless we the administrators make more tenants). Or if you have multiple subscriptions, pick the right one:
az account set --subscription 0bdb7994-618e-43ab-9dc4-e5510263d104
```

### b. Install Dependencies
```bash
# Step 2: Install Python goodies
pip install -r requirements.txt
```

If that fails or complains about missing packages, you might need to do some one-off installs like `pip-install streamlit azure-keyvault-secrets`. The `requirements.txt` should cover you, though. You got this.

### SSL Certificate (If your PostgreSQL Whines)
Some versions of Azure PostgreSQL demand a root cert. Check out the official Azure docs or do the quick fix:
1. Download the root cert chain (e.g., from Microsoft or DigiCert).
2. Convert `.cer` files to `.pem` (e.g. `openssl x509 -inform DER -in "root.cer" -out "root.pem" -outform PEM`).
3. In your code, point `sslrootcert` to the `.pem` file.

If you're looking for the specific way to do this, I downloaded the latest Azure Root Certificates from :
Microsoft’s root CA: https://www.microsoft.com/pkiops/certs/Microsoft%20RSA%20Root%20Certificate%20Authority%202017.cer
DigiCert Global Root G2: https://dl.cacerts.digicert.com/DigiCertGlobalRootG2.crt
DigiCert Global Root CA (for legacy compatibility): https://dl.cacerts.digicert.com/DigiCertGlobalRootCA.crt

Then, in the Downloads folder,
```bash
openssl x509 -inform DER -in "Microsoft RSA Root Certificate Authority 2017.cer" \
  -out microsoft_root_ca.pem -outform PEM

openssl x509 -inform DER \
  -in "DigiCertGlobalRootG2.crt" \
  -out digicert_global_root_g2.pem \
  -outform PEM

openssl x509 -inform DER \
  -in "DigiCertGlobalRootCA.crt" \
  -out digicert_global_root_ca.pem \
  -outform PEM

cat microsoft_root_ca.pem >> azure_root_chain.pem
cat digicert_global_root_g2.pem >> azure_root_chain.pem
cat digicert_global_root_ca.pem >> azure_root_chain.pem
```
This file, `azure_root_chain.pem`, is placed in and accessed via the root directory of this repository `tikaPOC`. Make sure to switch `sslrootcert="/Users/deangladish/tikaPOC/azure_root_chain.pem"` to the working directory `pwd`.

### d. Run the Application
Finally:
```bash
# Step 3: Launch it
streamlit run src/ui/streamlit_app.py
```

Pop open `http://localhost:8501`. Bam, chat with your Retrieval-Augmented Generation Agent right in your browser.

## 2. Technical Details

If you're the type who likes to see how the sausage is made, here's the skinny:

### a. Topic Data Pipeline
Pulls research topics from OpenAlex using cursor-based pagination.
- Cleans and normalizes the topics and keywords.
- Stores raw data + mappings in JSON.

### b. Database & Embeddings
PostgreSQL with `pgvector` for similarity searching.
Tables:
- `topics` (contains basic info)
- `keywords` (with 768-dim SciBERT embeddings)
- bridging table for many-to-many relationships
- Index: IVFFlat for speedy vector queries.
- Embeddings: SciBERT-based embeddings get auto-generated and stored.

### c. Topic Agent & Search
- Multi-turn query logic that considers conversation history.
- Rewrites queries on the fly for better context.
- Harnesses vector similarity search (pgvector).
- Excludes repeated or irrelevant topics.
- Ranks hits by similarity score so you get good recommendations.

### d. Chat Manager
- Manages the back-and-forth flow in your conversation.
- Taps into Azure OpenAI for GPT-4 magic.
- Has fallback paths if the OpenAI endpoints are flaky.
- Maintains session state so your context doesn't go poof.

### e. User Interface
- Built on **Streamlit** so you get a snappy, real-time chat experience.
- Auto-updates when you send messages or the agent has new thoughts.

### f. Cloud Infrastructure
- **Resource Group:** `tikabox`
**Services:**
- Azure PostgreSQL for topic/keyword storage
- Azure Key Vault for secrets & credentials
- Azure OpenAI Service (GPT-4) for query processing & generation
**Auth:** Driven by Azure's `DefaultAzureCredential` (which basically tries everything in your environment to figure out who you are).

That's the gist. Go forth and conquer some research topics with your brand-new Socratic RAG buddy. Ping us if you hit any snags--cloud configs can get spicy sometimes. Have fun!

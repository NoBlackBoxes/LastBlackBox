# lastblackbox.training

This folder contains the front- and back-end tools/code for creating and hosting the Last Black Box [website](https://lastblackbox.training).

----

## Security

Secure HTTP requests ("https") on port 443 must be handled by a secure socket (SSL). This will require SSL key/certificates. It is possible to generate these for a local server and tell your browser to accept them (despite being self-signed). However, for a public host, these must be signed by a certififcate authority.

- **Localhost**

  - Generate (and sign) the SSL certificate (lbb.pem) and private key (key.pem) using script

- **AWS host**

  - Generate the SSL certififcates using CertBot
  - Add a symlink to lbb.pem and key.pem in the LBBROOT
  - Make sure that HTTPS is enabled on AWS LightSail

----

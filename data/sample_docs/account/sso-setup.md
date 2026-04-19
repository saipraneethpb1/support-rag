# Set up SSO

Single sign-on is available on Team (Google, Microsoft) and Business
(SAML, Okta, OneLogin) plans.

## SAML SSO (Business plan)

1. In Flowpoint, open **Settings > Security > SSO** and copy the ACS URL
   and Entity ID.
2. In your identity provider, create a new SAML app with those values.
3. Paste the IdP metadata URL back into Flowpoint.
4. Click **Test connection**. A successful test shows your email from
   the IdP.
5. Enable **Require SSO** to force all members to sign in via SSO.

## Common errors

- **"Invalid audience"** — The Entity ID in your IdP doesn't match
  Flowpoint's. Copy it again from the SSO settings page.
- **"User not provisioned"** — SCIM isn't enabled or the user doesn't
  exist in Flowpoint yet. Either enable SCIM auto-provisioning or
  invite the user manually first.
- **Redirect loop after sign-in** — Your IdP is returning a NameID
  format Flowpoint doesn't recognize. Set NameID format to
  `emailAddress`.

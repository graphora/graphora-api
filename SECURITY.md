# Security Policy

## Reporting a Vulnerability

We take the security of Graphora seriously. If you discover a security vulnerability, please follow these steps:

### 1. Do Not Open a Public Issue

Please **do not** report security vulnerabilities through public GitHub issues, discussions, or pull requests.

### 2. Report Privately

Send your vulnerability report to: **support@graphora.io**

Include the following information:
- Type of vulnerability
- Full paths of source files related to the vulnerability
- Location of the affected source code (tag/branch/commit or direct URL)
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the vulnerability
- Any potential mitigations you've identified

### 3. What to Expect

- **Acknowledgment**: We'll acknowledge receipt within 48 hours
- **Updates**: We'll keep you informed about our progress
- **Timeline**: We aim to provide an initial assessment within 7 days
- **Fix**: Critical issues will be addressed immediately; others within 30 days
- **Disclosure**: We'll coordinate with you on public disclosure timing

### 4. Responsible Disclosure

We request that you:
- Allow us reasonable time to fix the vulnerability before public disclosure
- Avoid exploiting the vulnerability or sharing it with others
- Do not access, modify, or delete data that isn't yours
- Act in good faith and avoid privacy violations

### 5. Recognition

If you responsibly disclose a security issue:
- We'll acknowledge you in our security advisory (unless you prefer to remain anonymous)
- We may offer a bounty for critical vulnerabilities (case-by-case basis)

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| main    | :white_check_mark: |
| < 1.0   | :x:                |

We currently support only the latest version. Security fixes will be backported on a case-by-case basis for critical issues.

## Security Best Practices

When deploying Graphora API:

### Authentication & Authorization
- Implement strong authentication mechanisms
- Use API keys or OAuth tokens
- Validate user permissions for all operations
- Rotate credentials regularly

### Network Security
- Always use HTTPS in production
- Configure CORS appropriately
- Implement rate limiting per user/IP
- Use firewall rules to restrict access

### Input Validation
- Validate all user inputs
- Sanitize file uploads
- Limit file sizes and types
- Use Pydantic models for validation

### Database Security
- Use parameterized queries (no raw Cypher)
- Implement connection pooling with limits
- Use read-only connections where possible
- Enable Neo4j authentication
- Encrypt database connections

### API Security
- Implement request signing
- Use CSRF protection
- Set appropriate timeout values
- Log all security-relevant events
- Implement request size limits

### Secrets Management
- Use environment variables for all secrets
- Never commit `.env` files
- Use secret management services (AWS Secrets Manager, HashiCorp Vault)
- Rotate API keys regularly
- Use different keys per environment

### LLM Integration Security
- Validate LLM responses
- Implement output filtering
- Set token limits
- Monitor for prompt injection attempts
- Use structured outputs where possible

### File Processing
- Validate file types before processing
- Scan uploads for malware
- Use isolated environments for processing
- Implement file size limits
- Clean up temporary files

### Dependencies
- Regularly update dependencies (`uv sync --upgrade`)
- Monitor security advisories
- Use `pip-audit` or similar tools
- Review dependency licenses

### Monitoring & Logging
- Log authentication attempts
- Monitor for unusual patterns
- Implement alerting for security events
- Retain logs for audit purposes
- Protect log data

### Deployment
- Use minimal container images
- Scan images for vulnerabilities
- Run with non-root users
- Use security scanning in CI/CD
- Implement secrets scanning

## Common Vulnerabilities to Avoid

### Injection Attacks
- SQL/Cypher injection - Use parameterized queries
- Command injection - Validate and sanitize inputs
- Prompt injection - Validate LLM inputs/outputs

### Authentication Issues
- Broken authentication - Use established libraries
- Session fixation - Regenerate session IDs
- Credential stuffing - Implement rate limiting

### Data Exposure
- Sensitive data exposure - Encrypt at rest and in transit
- Excessive data exposure - Return only necessary fields
- Insufficient logging - Log security events

### Access Control
- Broken access control - Validate permissions
- IDOR - Use UUIDs, validate ownership
- Path traversal - Validate file paths

## Security Features

### Implemented
- Input validation with Pydantic
- CORS configuration
- Request timeouts
- Error handling (no sensitive data in errors)

### Recommended
- Rate limiting (implement in production)
- API key rotation
- Audit logging
- Security headers
- Request signing

## Incident Response

In case of a security incident:

1. **Contain**: Isolate affected systems
2. **Investigate**: Determine scope and impact
3. **Notify**: Inform affected users
4. **Remediate**: Deploy fixes
5. **Document**: Record lessons learned

## Security Updates

Security updates will be announced through:
- GitHub Security Advisories
- Release notes
- Email notifications (for registered users)
- Security mailing list

## Security Contacts

- **Security Issues**: support@graphora.io
- **General Questions**: See [SUPPORT.md](SUPPORT.md)
- **Enterprise Support**: support@graphora.io

## Compliance

Graphora API is designed to support:
- GDPR compliance (data processing)
- SOC 2 requirements (with proper configuration)
- HIPAA considerations (healthcare deployments)

*Note: Compliance certification is the responsibility of the deploying organization.*

---

**Thank you for helping keep Graphora secure!**

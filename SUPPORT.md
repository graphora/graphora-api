# Support

Thank you for using Graphora! This document outlines how to get help with the project.

## Documentation

- **README**: [README.md](README.md) - Quick start and overview
- **Contributing Guide**: [CONTRIBUTING.md](CONTRIBUTING.md) - How to contribute
- **API Documentation**: Available at `/api/v1/docs` when running the server

## Community Support

### GitHub Discussions
The best place for questions, ideas, and community discussion:
- **Q&A**: Ask and answer questions
- **Ideas**: Propose new features
- **Show and Tell**: Share what you've built
- **General**: Everything else

👉 [Start a Discussion](https://github.com/graphora/graphora-api/discussions)

### GitHub Issues
For bug reports and feature requests:
- **Bug Reports**: [Report a bug](https://github.com/graphora/graphora-api/issues/new?template=bug_report.md)
- **Feature Requests**: [Request a feature](https://github.com/graphora/graphora-api/issues/new?template=feature_request.md)

**Before creating an issue:**
- Search existing issues to avoid duplicates
- Provide as much detail as possible
- Include reproduction steps for bugs
- Add relevant logs and stack traces

### Community Chat
Join our community chat for real-time discussions:
- **Discord**: [Join Discord Server](https://discord.gg/graphora) *(coming soon)*
- **Slack**: [Join Slack Workspace](https://graphora.slack.com) *(coming soon)*

## Getting Help

### Common Questions

**Installation Issues**
- Make sure you have Python 3.11+ installed
- Install `uv`: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Run `uv sync` to install dependencies

**Configuration Issues**
- Copy `.env.sample` to `.env`
- Set up your LLM API keys (OpenAI, Anthropic, etc.)
- Configure Neo4j connection
- Set up Prefect for workflow management

**Database Connection Issues**
- Verify Neo4j is running
- Check connection credentials in `.env`
- Ensure network connectivity to database
- Check firewall rules

**API Issues**
- Server not starting: Check port 8000 is available
- CORS errors: Configure CORS settings in `main.py`
- Timeout errors: Adjust timeout settings in `.env`

### Troubleshooting Steps

1. **Check the logs** - Error messages usually point to the issue
2. **Verify configuration** - Double-check all environment variables
3. **Test dependencies** - Ensure Neo4j, LLM APIs are accessible
4. **Search existing issues** - Someone may have had the same problem
5. **Ask in Discussions** - Community members can help
6. **Create an issue** - If you've found a bug

### Getting Better Help

When asking for help, include:
- **Environment**: OS, Python version, uv version
- **Configuration**: Relevant `.env` settings (redact secrets!)
- **Steps to reproduce**: What you did
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happened
- **Logs**: Full error messages and stack traces
- **Code samples**: Minimal reproduction

## Commercial Support

### Enterprise Support
For production deployments and enterprise needs:
- Dedicated support channel
- SLA guarantees (99.9% uptime)
- Priority bug fixes
- Custom feature development
- Architecture review
- Performance tuning
- Training and onboarding
- 24/7 on-call support

📧 Contact: support@graphora.io

### Consulting Services
Professional services available:
- **Custom Integrations**: Connect to your data sources
- **Performance Optimization**: Scale to millions of documents
- **Custom Ontologies**: Domain-specific knowledge models
- **Migration Assistance**: Move from other platforms
- **Training Workshops**: Team training sessions
- **Architecture Design**: System design consultation

📧 Contact: sales@graphora.io

### Commercial Licensing
If you need to:
- Deploy as a closed-source SaaS platform
- Keep modifications proprietary
- Embed in a commercial product
- Get an OEM license for database vendors
- Offer Graphora as a managed service

We offer flexible commercial licensing options.

📧 Contact: support@graphora.io

### Database Vendor Partnerships
Special OEM licensing for database companies:
- Integrate Graphora into your platform
- White-label options available
- Co-marketing opportunities
- Technical partnership program

📧 Contact: sales@graphora.io

## Contributing

Want to contribute? We'd love your help!
- Read the [Contributing Guide](CONTRIBUTING.md)
- Check [good first issues](https://github.com/graphora/graphora-api/labels/good%20first%20issue)
- Join the community discussions
- Improve documentation
- Add tests
- Fix bugs

## Security Issues

**Do not** report security vulnerabilities in public issues.

See our [Security Policy](SECURITY.md) for how to report security issues privately.

📧 Security contact: support@graphora.io

## Resources

### Official Links
- **Website**: https://graphora.io *(coming soon)*
- **Documentation**: https://docs.graphora.io *(coming soon)*
- **API Reference**: https://api.graphora.io/docs *(coming soon)*
- **Blog**: https://blog.graphora.io *(coming soon)*
- **Twitter**: [@graphora](https://twitter.com/graphora) *(coming soon)*

### Related Repositories
- **Frontend**: [graphora/graphora-fe](https://github.com/graphora/graphora-fe)
- **Python Client**: [graphora/graphora-client](https://github.com/graphora/graphora-client)

### Learning Resources
- Tutorial videos *(coming soon)*
- Example projects *(coming soon)*
- Use case guides *(coming soon)*
- Webinars and workshops *(coming soon)*

### Integration Guides
- Neo4j setup
- LLM provider configuration
- Prefect workflow setup
- Cloud deployment guides

## Response Times

### Community Support (Free)
- GitHub Discussions: Best effort, typically 24-48 hours
- GitHub Issues: Triaged within 7 days
- Pull Requests: Reviewed within 14 days
- Critical bugs: Prioritized for next release

### Commercial Support (Paid)
- **Standard**: 8-hour response (business hours)
- **Premium**: 4-hour response (business hours)
- **Enterprise**: 1-hour response (24/7)
- **Critical incidents**: Immediate response (24/7)

## Language Support

Primary language: English

Community translations welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on translating documentation.

## FAQ

**Q: Which LLM providers are supported?**
A: OpenAI, Anthropic, Google Gemini, Mistral, and any OpenAI-compatible API.

**Q: Can I use local LLMs?**
A: Yes, via Ollama or any OpenAI-compatible local server.

**Q: What databases are supported?**
A: Currently Neo4j. Support for other graph databases is planned.

**Q: Is there a hosted version?**
A: Coming soon! Join the waitlist at graphora.io

**Q: Can I contribute new extractors?**
A: Yes! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Code of Conduct

All community interactions are governed by our [Code of Conduct](CODE_OF_CONDUCT.md). Please be respectful and inclusive.

---

**Need help?** Don't hesitate to ask! We're here to help you succeed with Graphora.

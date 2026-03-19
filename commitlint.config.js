module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        // Rule: 2 = error, 'always' = always check, and the list of allowed scopes
        'scope-enum': [
            2,
            'always',
            [
                'api',        // FastAPI
                'db',         // Database / PostgreSQL / SQL
                'docker',     // Dockerfile / docker-compose
                'ci',         // GitHub Actions
                'core',       // App core / Bash scripts
                'docs'        // Documentation
            ]
        ],
        // small letters only for scope
        'scope-case': [2, 'always', 'lower-case']
    }
};

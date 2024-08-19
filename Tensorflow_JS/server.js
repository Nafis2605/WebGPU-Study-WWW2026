const express = require('express');
const path = require('path');
const { createProxyMiddleware } = require('http-proxy-middleware');
const app = express();
const PORT = 5500;

// Middleware to add headers
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  console.log('Cross-Origin-Opener-Policy:', res.getHeader('Cross-Origin-Opener-Policy'));
  console.log('Cross-Origin-Embedder-Policy:', res.getHeader('Cross-Origin-Embedder-Policy'));
  next();
});

// Proxy setup to bypass CORS issues
app.use('/tfhub', createProxyMiddleware({
  target: 'https://tfhub.dev',
  changeOrigin: true,
  pathRewrite: {
      '^/tfhub': '', // Ensures the prefix is removed
  },
  onProxyRes: function(proxyRes, req, res) {
      proxyRes.headers['Access-Control-Allow-Origin'] = '*'; // Add CORS headers
  },
}));

app.get('/test', (req, res) => {
  res.send('Testing middleware');
})

// Serve static files from the current directory
app.use(express.static(path.join(__dirname, '/')));

app.listen(PORT, () => {
  console.log(`Server is running on http://localhost:${PORT}`);
  console.log(`Test cross-origin isolation at http://localhost:${PORT}/your-page.html`);
});

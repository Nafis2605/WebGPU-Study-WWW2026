const express = require('express');
const bodyParser = require('body-parser');

const app = express();
app.use(bodyParser.json());

app.post('/log', (req, res) => {
  console.log('Received data:', req.body);
  // Save data to a file or database
  res.status(200).json({ message: 'Data received' });
});

app.listen(3000, () => {
  console.log('Server running on http://localhost:3000');
});

from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras import layers, models

app = Flask(__name__)

@app.route('/configure_model')
def configure_model():
    # Get parameters from the URL
    layers_count = int(request.args.get('layers', 3))
    units = int(request.args.get('units', 64))
    activation = request.args.get('activation', 'relu')

    # Build a simple sequential model
    model = models.Sequential()
    for _ in range(layers_count):
        model.add(layers.Dense(units=units, activation=activation))
    model.add(layers.Dense(units=1, activation='sigmoid'))  # Output layer

    # Compile the model
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    # Return a JSON response with the model summary
    model_summary = []
    model.summary(print_fn=lambda x: model_summary.append(x))
    model_summary = "\n".join(model_summary)

    return jsonify({
        'model_summary': model_summary,
        'parameters': {
            'layers_count': layers_count,
            'units': units,
            'activation': activation,
        }
    })

if __name__ == '__main__':
    app.run(debug=True)

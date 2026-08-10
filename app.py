from flask import Flask, render_template, request, jsonify

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/location", methods=["POST"])
def receive_location():

    data = request.get_json()

    print("\n--- GPS DATA ---")
    print("Latitude :", data.get("latitude"))
    print("Longitude:", data.get("longitude"))
    print("Speed    :", data.get("speed"), "km/h")
    print("Accuracy :", data.get("accuracy"), "meters")
    print("Timestamp:", data.get("timestamp"))

    return jsonify({
        "success": True,
        "message": "GPS data received successfully"
    })


if __name__ == "__main__":
    app.run(debug=True)
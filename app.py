from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/route", methods=["POST"])
def get_route():

    data = request.get_json()

    start_lat = data.get("start_lat")
    start_lon = data.get("start_lon")

    end_lat = data.get("end_lat")
    end_lon = data.get("end_lon")

    if not all([
        start_lat,
        start_lon,
        end_lat,
        end_lon
    ]):
        return jsonify({
            "success": False,
            "message": "Missing coordinates"
        }), 400


    # OSRM uses:
    # longitude,latitude
    coordinates = (
        f"{start_lon},{start_lat};"
        f"{end_lon},{end_lat}"
    )

    url = (
        "https://router.project-osrm.org/route/v1/driving/"
        + coordinates
    )

    params = {
        "overview": "full",
        "geometries": "geojson"
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        osrm_data = response.json()

        if osrm_data.get("code") != "Ok":
            return jsonify({
                "success": False,
                "message": "Route not found"
            }), 404


        route = osrm_data["routes"][0]

        return jsonify({
            "success": True,
            "distance": route["distance"],
            "duration": route["duration"],
            "geometry": route["geometry"]
        })


    except requests.RequestException as error:

        print("OSRM Error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to connect to routing service"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)
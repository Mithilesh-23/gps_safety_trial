from flask import Flask, render_template, request, jsonify
import requests
import math

app = Flask(__name__)


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():
    return render_template("index.html")


# =========================================================
# SEARCH LOCATION
# Nominatim / OpenStreetMap
# =========================================================

@app.route("/api/search-location", methods=["GET"])
def search_location():

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify({
            "success": False,
            "message": "Location search is required"
        }), 400

    url = "https://nominatim.openstreetmap.org/search"

    params = {
        "q": query,
        "format": "json",
        "limit": 5
    }

    headers = {
        "User-Agent": "WomenSafetyGPSPrototype/1.0"
    }

    try:

        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=10
        )

        response.raise_for_status()

        results = response.json()

        locations = []

        for result in results:

            locations.append({
                "name": result["display_name"],
                "latitude": float(result["lat"]),
                "longitude": float(result["lon"])
            })

        return jsonify({
            "success": True,
            "locations": locations
        })

    except requests.RequestException as error:

        print("Nominatim error:", error)

        return jsonify({
            "success": False,
            "message": "Unable to search location"
        }), 500


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):

    earth_radius = 6371.0

    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)

    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )

    a = (
        math.sin(delta_lat / 2) ** 2
        +
        math.cos(lat1_rad)
        *
        math.cos(lat2_rad)
        *
        math.sin(delta_lon / 2) ** 2
    )

    c = 2 * math.atan2(
        math.sqrt(a),
        math.sqrt(1 - a)
    )

    return earth_radius * c


# =========================================================
# GENERATE INTERMEDIATE WAYPOINTS
# =========================================================

def generate_waypoints(
    start_lat,
    start_lon,
    end_lat,
    end_lon
):

    lat_difference = (
        end_lat - start_lat
    )

    lon_difference = (
        end_lon - start_lon
    )

    length = math.sqrt(
        lat_difference ** 2
        +
        lon_difference ** 2
    )

    if length == 0:
        return []


    # Perpendicular direction

    perpendicular_lat = (
        -lon_difference / length
    )

    perpendicular_lon = (
        lat_difference / length
    )


    # Midpoint

    middle_lat = (
        start_lat + end_lat
    ) / 2

    middle_lon = (
        start_lon + end_lon
    ) / 2


    # Waypoint deviation

    deviation = 0.015


    waypoints = [

        # Left route
        (
            middle_lat
            +
            perpendicular_lat * deviation,

            middle_lon
            +
            perpendicular_lon * deviation
        ),

        # Right route
        (
            middle_lat
            -
            perpendicular_lat * deviation,

            middle_lon
            -
            perpendicular_lon * deviation
        ),

        # Further left
        (
            middle_lat
            +
            perpendicular_lat * deviation * 2,

            middle_lon
            +
            perpendicular_lon * deviation * 2
        ),

        # Further right
        (
            middle_lat
            -
            perpendicular_lat * deviation * 2,

            middle_lon
            -
            perpendicular_lon * deviation * 2
        )
    ]

    return waypoints


# =========================================================
# GET ROUTE FROM OSRM
# =========================================================

def get_osrm_route(coordinates):

    # OSRM requires:
    #
    # longitude,latitude;
    # longitude,latitude

    coordinate_string = ";".join(
        [
            f"{longitude},{latitude}"
            for latitude, longitude in coordinates
        ]
    )


    url = (
        "https://router.project-osrm.org/"
        "route/v1/driving/"
        + coordinate_string
    )


    params = {
        "overview": "full",
        "geometries": "geojson",
        "steps": "false"
    }


    try:

        response = requests.get(
            url,
            params=params,
            timeout=15
        )

        response.raise_for_status()

        data = response.json()


        if data.get("code") != "Ok":

            return None


        routes = data.get(
            "routes",
            []
        )


        if not routes:

            return None


        return routes[0]


    except requests.RequestException as error:

        print(
            "OSRM error:",
            error
        )

        return None


# =========================================================
# CHECK ROUTE SIMILARITY
# =========================================================

def routes_are_similar(
    route1,
    route2
):

    distance1 = route1["distance"]

    distance2 = route2["distance"]


    average_distance = (
        distance1 + distance2
    ) / 2


    if average_distance == 0:

        return True


    difference = abs(
        distance1 - distance2
    )


    percentage_difference = (
        difference / average_distance
    ) * 100


    # If distances differ by less than 3%,
    # treat them as potentially duplicate
    # candidate routes.

    if percentage_difference < 3:

        return True


    return False


# =========================================================
# ROUTE API
# =========================================================

@app.route(
    "/api/route",
    methods=["POST"]
)
def get_multiple_routes():

    data = request.get_json()


    if not data:

        return jsonify({
            "success": False,
            "message": "Request body is required"
        }), 400


    # Starting location

    start_lat = data.get(
        "start_lat"
    )

    start_lon = data.get(
        "start_lon"
    )


    # Destination

    end_lat = data.get(
        "end_lat"
    )

    end_lon = data.get(
        "end_lon"
    )


    # =====================================================
    # VALIDATE
    # =====================================================

    if None in [
        start_lat,
        start_lon,
        end_lat,
        end_lon
    ]:

        return jsonify({

            "success": False,

            "message":
                "Starting location and destination are required"

        }), 400


    try:

        start_lat = float(start_lat)
        start_lon = float(start_lon)

        end_lat = float(end_lat)
        end_lon = float(end_lon)

    except (
        TypeError,
        ValueError
    ):

        return jsonify({

            "success": False,

            "message":
                "Coordinates must be numbers"

        }), 400


    # =====================================================
    # CHECK SAME LOCATION
    # =====================================================

    straight_distance = calculate_distance(
        start_lat,
        start_lon,
        end_lat,
        end_lon
    )


    if straight_distance < 0.05:

        return jsonify({

            "success": False,

            "message":
                "Starting location and destination are too close"

        }), 400


    # =====================================================
    # CANDIDATE ROUTES
    # =====================================================

    candidate_routes = []


    # =====================================================
    # ROUTE 1
    # DIRECT ROUTE
    # =====================================================

    direct_route = get_osrm_route(
        [
            (
                start_lat,
                start_lon
            ),

            (
                end_lat,
                end_lon
            )
        ]
    )


    if direct_route is not None:

        candidate_routes.append(
            direct_route
        )


    # =====================================================
    # GENERATE WAYPOINTS
    # =====================================================

    waypoints = generate_waypoints(

        start_lat,
        start_lon,

        end_lat,
        end_lon

    )


    # =====================================================
    # CREATE ALTERNATIVE ROUTES
    # =====================================================

    for waypoint in waypoints:

        waypoint_route = get_osrm_route(
            [
                (
                    start_lat,
                    start_lon
                ),

                waypoint,

                (
                    end_lat,
                    end_lon
                )
            ]
        )


        if waypoint_route is None:

            continue


        # Check duplicate

        duplicate = False


        for existing_route in candidate_routes:

            if routes_are_similar(
                existing_route,
                waypoint_route
            ):

                duplicate = True

                break


        if not duplicate:

            candidate_routes.append(
                waypoint_route
            )


    # =====================================================
    # SORT BY DISTANCE
    # =====================================================

    candidate_routes.sort(
        key=lambda route:
            route["distance"]
    )


    # =====================================================
    # LIMIT ROUTES
    # =====================================================

    # For the prototype, keep
    # maximum 4 routes.

    candidate_routes = (
        candidate_routes[:4]
    )


    # =====================================================
    # FORMAT RESPONSE
    # =====================================================

    routes = []


    for index, route in enumerate(
        candidate_routes
    ):

        distance_km = (
            route["distance"]
            / 1000
        )


        duration_minutes = (
            route["duration"]
            / 60
        )


        routes.append({

            "route_id":
                index + 1,

            "distance_km":
                round(
                    distance_km,
                    2
                ),

            "duration_minutes":
                round(
                    duration_minutes,
                    2
                ),

            "geometry":
                route["geometry"]

        })


    # =====================================================
    # RESPONSE
    # =====================================================

    return jsonify({

        "success": True,

        "route_count":
            len(routes),

        "routes":
            routes

    })


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
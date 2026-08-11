from flask import Flask, render_template, request, jsonify
import requests
import math
import pandas as pd
import os


app = Flask(__name__)


# =========================================================
# FILES
# =========================================================

SAFETY_FILE = "safety_data.csv"

IMPORTANT_PLACES_FILE = "important_places.csv"

SAFETY_RADIUS_KM = 0.5


# =========================================================
# LOAD SAFETY DATA
# =========================================================

def load_safety_data():

    try:

        df = pd.read_csv(SAFETY_FILE)

        required = [
            "latitude",
            "longitude"
        ]

        missing = [
            column
            for column in required
            if column not in df.columns
        ]

        if missing:

            print(
                "Safety dataset missing:",
                missing
            )

            return pd.DataFrame()


        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )


        df = df.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )


        print(
            "Safety data loaded:",
            len(df),
            "records"
        )


        return df


    except FileNotFoundError:

        print(
            "WARNING: safety_data.csv not found"
        )

        return pd.DataFrame()


    except Exception as error:

        print(
            "Safety dataset error:",
            error
        )

        return pd.DataFrame()


# =========================================================
# LOAD IMPORTANT PLACES
# =========================================================

def load_important_places():

    try:

        df = pd.read_csv(
            IMPORTANT_PLACES_FILE
        )


        required = [
            "name",
            "type",
            "latitude",
            "longitude"
        ]


        missing = [

            column

            for column in required

            if column not in df.columns

        ]


        if missing:

            print(
                "Important places missing:",
                missing
            )

            return pd.DataFrame()


        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )


        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )


        df = df.dropna(

            subset=[
                "latitude",
                "longitude"
            ]

        )


        print(
            "Important places loaded:",
            len(df),
            "records"
        )


        return df


    except FileNotFoundError:

        print(
            "WARNING: important_places.csv not found"
        )

        return pd.DataFrame()


    except Exception as error:

        print(
            "Important places error:",
            error
        )

        return pd.DataFrame()


# Load at startup

safety_data = load_safety_data()

important_places = load_important_places()

# =========================================================
# LOAD AREA RISK DATA
# =========================================================

AREA_RISK_FILE = "area_risk.csv"

MAX_RISK_DISTANCE_KM = 0.5


def load_area_risk():

    try:

        df = pd.read_csv(AREA_RISK_FILE)

        required_columns = [
            "police_station",
            "latitude",
            "longitude",
            "risk_score"
        ]

        for column in required_columns:

            if column not in df.columns:

                print(
                    "Missing column:",
                    column
                )

                return pd.DataFrame()


        df["latitude"] = pd.to_numeric(
            df["latitude"],
            errors="coerce"
        )

        df["longitude"] = pd.to_numeric(
            df["longitude"],
            errors="coerce"
        )

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
            errors="coerce"
        )


        df = df.dropna(
            subset=[
                "latitude",
                "longitude",
                "risk_score"
            ]
        )


        print(
            "Area risk data loaded:",
            len(df),
            "records"
        )


        return df


    except FileNotFoundError:

        print(
            "WARNING: area_risk.csv not found"
        )

        return pd.DataFrame()


    except Exception as error:

        print(
            "Area risk loading error:",
            error
        )

        return pd.DataFrame()


area_risk_data = load_area_risk()

# =========================================================
# CALCULATE DISTANCE WEIGHT
# =========================================================

def calculate_distance_weight(
    distance_km
):

    if distance_km <= 0.1:

        return 1.0


    elif distance_km <= 0.3:

        return 0.7


    elif distance_km <= 0.5:

        return 0.3


    else:

        return 0.0


# =========================================================
# CALCULATE ROUTE RISK
# =========================================================

def calculate_route_risk(
    route_geometry
):

    if area_risk_data.empty:

        return {
            "route_risk": 0,
            "safety_score": 100,
            "risk_points": []
        }


    coordinates = route_geometry.get(
        "coordinates",
        []
    )


    if not coordinates:

        return {
            "route_risk": 0,
            "safety_score": 100,
            "risk_points": []
        }


    risk_points = []


    # =====================================================
    # CHECK EVERY RISK POINT
    # =====================================================

    for _, row in area_risk_data.iterrows():

        try:

            risk_lat = float(
                row["latitude"]
            )

            risk_lon = float(
                row["longitude"]
            )

            risk_score = float(
                row["risk_score"]
            )

        except (
            TypeError,
            ValueError
        ):

            continue


        # -------------------------------------------------
        # Find the nearest point of this risk location
        # to the route geometry.
        # -------------------------------------------------

        minimum_distance = float(
            "inf"
        )


        for coordinate in coordinates:

            if not coordinate or len(coordinate) < 2:

                continue


            route_lon = float(
                coordinate[0]
            )

            route_lat = float(
                coordinate[1]
            )


            distance = calculate_distance(

                route_lat,
                route_lon,

                risk_lat,
                risk_lon

            )


            if distance < minimum_distance:

                minimum_distance = distance


        # -------------------------------------------------
        # Ignore risk locations farther than 500 metres.
        # -------------------------------------------------

        if minimum_distance > MAX_RISK_DISTANCE_KM:

            continue


        # -------------------------------------------------
        # Normalize risk score.
        #
        # Dataset may contain either:
        #     0 - 1
        # or:
        #     0 - 100
        # -------------------------------------------------

        if 0 <= risk_score <= 1:

            risk_score *= 100


        risk_score = max(
            0,
            min(
                100,
                risk_score
            )
        )


        # -------------------------------------------------
        # Distance weight
        #
        # 0 - 100 m  -> 1.0
        # 100-300 m  -> 0.7
        # 300-500 m  -> 0.3
        # -------------------------------------------------

        weight = calculate_distance_weight(
            minimum_distance
        )


        weighted_risk = (
            risk_score * weight
        )


        risk_points.append({

            "police_station":
                str(
                    row.get(
                        "police_station",
                        row.get(
                            "location_name",
                            "Risk area"
                        )
                    )
                ),

            "latitude":
                risk_lat,

            "longitude":
                risk_lon,

            "distance_km":
                round(
                    minimum_distance,
                    3
                ),

            "risk_score":
                round(
                    risk_score,
                    2
                ),

            "weight":
                weight,

            "weighted_risk":
                round(
                    weighted_risk,
                    2
                )

        })


    # =====================================================
    # NO RISK POINTS
    # =====================================================

    if not risk_points:

        return {

            "route_risk":
                0,

            "safety_score":
                100,

            "risk_points":
                []

        }


    # =====================================================
    # SEVERITY COMPONENT
    # =====================================================

    total_weighted_risk = sum(

        point["weighted_risk"]

        for point in risk_points

    )


    total_weight = sum(

        point["weight"]

        for point in risk_points

    )


    if total_weight > 0:

        average_risk = (

            total_weighted_risk
            /
            total_weight

        )

    else:

        average_risk = 0


    average_risk = max(
        0,
        min(
            100,
            average_risk
        )
    )


    # =====================================================
    # RISK-DENSITY COMPONENT
    # =====================================================
    #
    # The old calculation used only the average severity.
    # Therefore:
    #
    # 111 risk points + average risk 40
    # and
    # 10 risk points  + average risk 40
    #
    # could receive almost the same score.
    #
    # This component makes the number of nearby risk
    # locations matter as well.
    #
    # Exponential saturation prevents the count from
    # making the score exceed 100.
    # =====================================================

    risk_point_count = len(
        risk_points
    )


    density_factor = (
        1
        -
        math.exp(
            -risk_point_count / 60.0
        )
    )


    density_risk = (
        density_factor * 100
    )


    # =====================================================
    # FINAL ROUTE RISK
    # =====================================================
    #
    # 70% -> severity of nearby risk locations
    # 30% -> density/number of nearby risk locations
    # =====================================================

    route_risk = (

        average_risk * 0.70

        +

        density_risk * 0.30

    )


    route_risk = max(
        0,
        min(
            100,
            route_risk
        )
    )


    # =====================================================
    # SAFETY SCORE
    # =====================================================

    safety_score = (
        100
        -
        route_risk
    )


    safety_score = max(
        0,
        min(
            100,
            safety_score
        )
    )


    # =====================================================
    # DEBUG INFORMATION
    # =====================================================

    print(
        "Route risk calculation:",
        "points =",
        risk_point_count,
        "| average risk =",
        round(
            average_risk,
            2
        ),
        "| density risk =",
        round(
            density_risk,
            2
        ),
        "| route risk =",
        round(
            route_risk,
            2
        ),
        "| safety score =",
        round(
            safety_score,
            2
        )
    )


    # =====================================================
    # RETURN
    # =====================================================

    return {

        "route_risk":
            round(
                route_risk,
                2
            ),

        "safety_score":
            round(
                safety_score,
                2
            ),

        "risk_points":
            risk_points

    }


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


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


    lat1_rad = math.radians(
        lat1
    )

    lat2_rad = math.radians(
        lat2
    )


    delta_lat = math.radians(
        lat2 - lat1
    )

    delta_lon = math.radians(
        lon2 - lon1
    )


    a = (

        math.sin(
            delta_lat / 2
        ) ** 2

        +

        math.cos(lat1_rad)

        *

        math.cos(lat2_rad)

        *

        math.sin(
            delta_lon / 2
        ) ** 2

    )


    c = 2 * math.atan2(

        math.sqrt(a),

        math.sqrt(1 - a)

    )


    return earth_radius * c


# =========================================================
# LOCAL PLACE SEARCH
# =========================================================

def search_local_places(query):

    if important_places.empty:

        return []


    query = query.lower().strip()


    if not query:

        return []


    results = []


    for _, row in important_places.iterrows():

        name = str(
            row["name"]
        ).lower()

        place_type = str(
            row["type"]
        ).lower()


        # Search in name or type

        if (

            query in name

            or

            query in place_type

        ):

            results.append({

                "name":
                    str(row["name"]),

                "type":
                    str(row["type"]),

                "latitude":
                    float(row["latitude"]),

                "longitude":
                    float(row["longitude"]),

                "source":
                    "local_dataset"

            })


    return results


# =========================================================
# SEARCH NOMINATIM
# =========================================================

def search_nominatim(query):

    url = (
        "https://nominatim.openstreetmap.org/"
        "search"
    )


    params = {

        "q": query,

        "format": "jsonv2",

        "limit": 5

    }


    headers = {

        "User-Agent":
            "WomenSafetyGPSPrototype/1.0"

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

                "name":
                    result.get(
                        "display_name",
                        query
                    ),

                "type":
                    "nominatim",

                "latitude":
                    float(
                        result["lat"]
                    ),

                "longitude":
                    float(
                        result["lon"]
                    ),

                "source":
                    "nominatim"

            })


        return locations


    except Exception as error:

        print(
            "Nominatim error:",
            error
        )

        return []


# =========================================================
# SEARCH LOCATION
# =========================================================

@app.route(
    "/api/search-location",
    methods=["GET"]
)
def search_location():

    query = request.args.get(
        "q",
        ""
    ).strip()


    if not query:

        return jsonify({

            "success": False,

            "message":
                "Location search is required"

        }), 400


    # =====================================================
    # FIRST: SEARCH OUR OWN DATASET
    # =====================================================

    local_results = search_local_places(
        query
    )


    if local_results:

        print(
            "Local dataset match:",
            query
        )


        return jsonify({

            "success": True,

            "source":
                "local_dataset",

            "locations":
                local_results

        })


    # =====================================================
    # SECOND: NOMINATIM FALLBACK
    # =====================================================

    print(
        "Local place not found. "
        "Using Nominatim:",
        query
    )


    nominatim_results = search_nominatim(
        query
    )


    if not nominatim_results:

        return jsonify({

            "success": False,

            "message":
                "No locations found"

        }), 404


    return jsonify({

        "success": True,

        "source":
            "nominatim",

        "locations":
            nominatim_results

    })


# =========================================================
# GENERATE WAYPOINTS
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


    perpendicular_lat = (
        -lon_difference / length
    )

    perpendicular_lon = (
        lat_difference / length
    )


    middle_lat = (
        start_lat + end_lat
    ) / 2


    middle_lon = (
        start_lon + end_lon
    ) / 2


    deviation = 0.015


    return [

        (

            middle_lat
            +
            perpendicular_lat
            * deviation,

            middle_lon
            +
            perpendicular_lon
            * deviation

        ),

        (

            middle_lat
            -
            perpendicular_lat
            * deviation,

            middle_lon
            -
            perpendicular_lon
            * deviation

        ),

        (

            middle_lat
            +
            perpendicular_lat
            * deviation
            * 2,

            middle_lon
            +
            perpendicular_lon
            * deviation
            * 2

        ),

        (

            middle_lat
            -
            perpendicular_lat
            * deviation
            * 2,

            middle_lon
            -
            perpendicular_lon
            * deviation
            * 2

        )

    ]


# =========================================================
# OSRM ROUTE
# =========================================================

def get_osrm_route(
    coordinates
):

    coordinate_string = ";".join(

        [

            f"{longitude},{latitude}"

            for latitude, longitude
            in coordinates

        ]

    )


    url = (

        "https://router.project-osrm.org/"
        "route/v1/driving/"
        +
        coordinate_string

    )


    params = {

        "overview":
            "full",

        "geometries":
            "geojson",

        "steps":
            "false"

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


    except Exception as error:

        print(
            "OSRM error:",
            error
        )

        return None


# =========================================================
# ROUTE SIMILARITY
# =========================================================

def routes_are_similar(

    route1,
    route2

):

    distance1 = route1["distance"]

    distance2 = route2["distance"]


    average = (
        distance1 + distance2
    ) / 2


    if average == 0:

        return True


    difference = abs(
        distance1 - distance2
    )


    percentage = (
        difference / average
    ) * 100


    return percentage < 3


# =========================================================
# FIND SAFETY DATA NEAR ROUTE
# =========================================================

def get_safety_data_for_route(

    route_geometry,

    radius_km=SAFETY_RADIUS_KM

):

    if safety_data.empty:

        return []


    coordinates = route_geometry.get(
        "coordinates",
        []
    )


    nearby_points = []


    for point in coordinates:

        route_lon = point[0]

        route_lat = point[1]


        for _, row in safety_data.iterrows():

            try:

                data_lat = float(
                    row["latitude"]
                )

                data_lon = float(
                    row["longitude"]
                )

            except Exception:

                continue


            distance = calculate_distance(

                route_lat,
                route_lon,

                data_lat,
                data_lon

            )


            if distance <= radius_km:

                safety_point = {

                    "latitude":
                        data_lat,

                    "longitude":
                        data_lon,

                    "distance_from_route_km":
                        round(
                            distance,
                            3
                        )

                }


                for column in safety_data.columns:

                    if column in [
                        "latitude",
                        "longitude"
                    ]:

                        continue


                    value = row[column]


                    if pd.isna(value):

                        value = None

                    elif isinstance(
                        value,
                        (int, float)
                    ):

                        value = float(
                            value
                        )

                    else:

                        value = str(
                            value
                        )


                    safety_point[column] = value


                nearby_points.append(
                    safety_point
                )


    # Remove duplicate points

    unique_points = []

    seen = set()


    for point in nearby_points:

        key = (

            round(
                point["latitude"],
                6
            ),

            round(
                point["longitude"],
                6
            )

        )


        if key not in seen:

            seen.add(key)

            unique_points.append(
                point
            )


    return unique_points


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

            "message":
                "Request body is required"

        }), 400


    start_lat = data.get(
        "start_lat"
    )

    start_lon = data.get(
        "start_lon"
    )

    end_lat = data.get(
        "end_lat"
    )

    end_lon = data.get(
        "end_lon"
    )


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

        start_lat = float(
            start_lat
        )

        start_lon = float(
            start_lon
        )

        end_lat = float(
            end_lat
        )

        end_lon = float(
            end_lon
        )

    except (

        TypeError,
        ValueError

    ):

        return jsonify({

            "success": False,

            "message":
                "Coordinates must be numbers"

        }), 400


    # Same location check

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


    candidate_routes = []


    # =====================================================
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


    if direct_route:

        candidate_routes.append(
            direct_route
        )


    # =====================================================
    # ALTERNATIVE ROUTES
    # =====================================================

    waypoints = generate_waypoints(

        start_lat,
        start_lon,

        end_lat,
        end_lon

    )


    for waypoint in waypoints:

        route = get_osrm_route(

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


        if route is None:

            continue


        # Keep every valid waypoint route for the prototype.
        # We limit the final list to four routes below.
        candidate_routes.append(
            route
        )


    # Sort by distance

    candidate_routes.sort(

        key=lambda route:
            route["distance"]

    )


    # Maximum 4 routes

    candidate_routes = candidate_routes[:4]


    # Debug: verify how many routes were generated.
    print("\n================ ROUTE DEBUG ================")
    print(
        "Generated waypoints:",
        len(waypoints)
    )
    print(
        "Candidate routes:",
        len(candidate_routes)
    )

    for debug_index, debug_route in enumerate(
        candidate_routes,
        start=1
    ):

        print(
            "Route",
            debug_index,
            "| Distance:",
            round(
                debug_route["distance"] / 1000,
                2
            ),
            "km",
            "| Duration:",
            round(
                debug_route["duration"] / 60,
                2
            ),
            "min"
        )

    print("=============================================\n")


    routes = []


    # =====================================================
    # FORMAT ROUTES
    # =====================================================

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


        # Find safety points

        nearby_safety_data = (

            get_safety_data_for_route(

                route["geometry"],

                SAFETY_RADIUS_KM

            )

        )

        route_risk_data = calculate_route_risk(
            route["geometry"]
        )

        print(
            "Route",
            index + 1,
            "| risk points:",
            len(
                route_risk_data["risk_points"]
            ),
            "| route risk:",
            route_risk_data["route_risk"],
            "| safety score:",
            route_risk_data["safety_score"]
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
                    route["geometry"],

                "safety_data":
                    nearby_safety_data,

                "safety_point_count":
                    len(
                        nearby_safety_data
                    ),

                "route_risk":
                    route_risk_data[
                        "route_risk"
                    ],

                "safety_score":
                    route_risk_data[
                        "safety_score"
                    ],

                "risk_points":
                    route_risk_data[
                        "risk_points"
                    ]

            })


    # =====================================================
    # FIND SAFEST ROUTE
    # =====================================================

    if routes:

        safest_route = max(
            routes,
            key=lambda route:
                route.get(
                    "safety_score",
                    0
                )
        )

        safest_route_id = (
            safest_route["route_id"]
        )

    else:

        safest_route_id = None


    # =====================================================
    # RESPONSE
    # =====================================================

    return jsonify({

        "success": True,

        "route_count":
            len(routes),

        # This identifies the safest route only.
        # It does NOT remove the other routes.
        "safest_route_id":
            safest_route_id,

        # Return ALL generated routes.
        "routes":
            routes

    })



# =========================================================
# IMPORTANT PLACES API
# =========================================================

@app.route(
    "/api/important-places",
    methods=["GET"]
)
def get_important_places():

    if important_places.empty:

        return jsonify({

            "success": False,

            "count": 0,

            "data": []

        })


    records = []


    for _, row in important_places.iterrows():

        records.append({

            "name":
                str(row["name"]),

            "type":
                str(row["type"]),

            "latitude":
                float(row["latitude"]),

            "longitude":
                float(row["longitude"])

        })


    return jsonify({

        "success": True,

        "count":
            len(records),

        "data":
            records

    })


# =========================================================
# SAFETY DATA API
# =========================================================
@app.route("/api/safety-data", methods=["GET"])
def get_safety_data():

    try:

        # Reload the same dataset used by route-risk calculation
        df = pd.read_csv("area_risk.csv")

        # Convert numeric fields
        numeric_columns = [
            "latitude",
            "longitude",
            "crime_risk",
            "lighting_level",
            "crowd_density",
            "traffic_level",
            "police_presence",
            "risk_score"
        ]

        for column in numeric_columns:

            if column in df.columns:

                df[column] = pd.to_numeric(
                    df[column],
                    errors="coerce"
                )

        # Remove records without coordinates
        df = df.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )

        # Replace NaN values with None for valid JSON
        df = df.where(
            pd.notnull(df),
            None
        )

        # Convert dataframe to JSON-compatible records
        safety_points = df.to_dict(
            orient="records"
        )

        print(
            f"Sending {len(safety_points)} safety points to frontend"
        )

        return jsonify({

            "success": True,

            "count": len(safety_points),

            "data": safety_points

        })

    except FileNotFoundError:

        return jsonify({

            "success": False,

            "error": "area_risk.csv not found",

            "data": []

        }), 404

    except Exception as error:

        print(
            "Safety data error:",
            error
        )

        return jsonify({

            "success": False,

            "error": str(error),

            "data": []

        }), 500

# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    app.run(
        debug=True
    )
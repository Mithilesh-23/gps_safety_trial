from flask import Flask, render_template, request, jsonify
import requests
import math
import pandas as pd
import os
from datetime import datetime

from ml_risk_predictor import predict_risk


app = Flask(__name__)


# =========================================================
# FILES
# =========================================================

SAFETY_FILE = "amravati_safety_data.csv"

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
            "longitude",
            "risk_score"
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

        df["risk_score"] = pd.to_numeric(
            df["risk_score"],
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

        print(
            "Safety data columns:",
            list(df.columns)
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
# UNIFIED SAFETY DATA FOR ROUTE RISK
# =========================================================
#
# We intentionally use the same dataset as the safety points
# shown on the map. This keeps the current route mechanism
# unchanged while making the safety score and displayed
# safety points use one source of truth.
#
# The unified CSV contains:
# latitude
# longitude
# risk_score
# location_name
# area_name
# place_type
# crime_risk
# lighting_level
# crowd_density
# traffic_level
# police_presence
# risk_level
# data_status
#
# =========================================================

MAX_RISK_DISTANCE_KM = SAFETY_RADIUS_KM

area_risk_data = safety_data.copy()

if area_risk_data.empty:
    print(
        "Unified safety data is empty."
    )
else:
    required_risk_columns = [
        "latitude",
        "longitude",
        "risk_score"
    ]

    missing_risk_columns = [
        column
        for column in required_risk_columns
        if column not in area_risk_data.columns
    ]

    if missing_risk_columns:
        print(
            "Unified safety dataset missing risk columns:",
            missing_risk_columns
        )

        area_risk_data = pd.DataFrame()

    else:
        area_risk_data["risk_score"] = pd.to_numeric(
            area_risk_data["risk_score"],
            errors="coerce"
        )

        area_risk_data = area_risk_data.dropna(
            subset=[
                "latitude",
                "longitude",
                "risk_score"
            ]
        )

        print(
            "Unified safety data used for route risk:",
            len(area_risk_data),
            "records"
        )

        print(
            "Unified safety columns:",
            list(area_risk_data.columns)
        )


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

        except (
            TypeError,
            ValueError
        ):

            continue


        # -------------------------------------------------
        # Find nearest point of this risk location
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
        # Ignore locations farther than safety radius.
        # -------------------------------------------------

        if minimum_distance > MAX_RISK_DISTANCE_KM:

            continue


        # =================================================
        # NORMALIZE HELPER
        # =================================================

        def normalize(value):

            try:

                value = float(value)

            except (
                TypeError,
                ValueError
            ):

                return 0.0


            return max(
                0.0,
                min(
                    100.0,
                    value
                )
            )


        # =================================================
        # READ MULTI-FACTOR DATA
        # =================================================

        crime_risk = normalize(
            row.get(
                "crime_risk",
                0
            )
        )

        lighting_level = normalize(
            row.get(
                "lighting_level",
                0
            )
        )

        crowd_density = normalize(
            row.get(
                "crowd_density",
                0
            )
        )

        traffic_level = normalize(
            row.get(
                "traffic_level",
                0
            )
        )

        police_presence = normalize(
            row.get(
                "police_presence",
                0
            )
        )

        existing_risk = normalize(
            row.get(
                "risk_score",
                0
            )
        )


        # =================================================
        # CONVERT SAFETY-POSITIVE FACTORS TO RISK
        # =================================================
        #
        # Higher lighting = safer
        # Higher police presence = safer
        #
        # We therefore convert them to risk values.
        #
        # Crowd density is kept as a risk factor in this
        # model because very high density can increase
        # congestion/exposure in the current prototype.
        # =================================================

        lighting_risk = (
            100.0 - lighting_level
        )

        police_risk = (
            100.0 - police_presence
        )

        crowd_risk = crowd_density


        # =================================================
        # MULTI-FACTOR POINT RISK
        # =================================================
        #
        # Crime Risk       = 30%
        # Lighting Risk    = 20%
        # Crowd Density    = 15%
        # Traffic Level    = 10%
        # Police Risk      = 15%
        # Existing Risk    = 10%
        #
        # Total            = 100%
        # =================================================

        point_risk = (

            crime_risk * 0.30

            +

            lighting_risk * 0.20

            +

            crowd_risk * 0.15

            +

            traffic_level * 0.10

            +

            police_risk * 0.15

            +

            existing_risk * 0.10

        )


        point_risk = max(
            0.0,
            min(
                100.0,
                point_risk
            )
        )


        # =================================================
        # EXISTING DISTANCE WEIGHTING
        # =================================================

        weight = calculate_distance_weight(
            minimum_distance
        )


        weighted_risk = (
            point_risk * weight
        )


        # =================================================
        # STORE ROUTE RISK POINT
        # =================================================

        risk_points.append({

            "location_name":
                str(
                    row.get(
                        "location_name",
                        row.get(
                            "police_station",
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
                    existing_risk,
                    2
                ),

            "calculated_point_risk":
                round(
                    point_risk,
                    2
                ),

            "weight":
                weight,

            "weighted_risk":
                round(
                    weighted_risk,
                    2
                ),

            "crime_risk":
                round(
                    crime_risk,
                    2
                ),

            "lighting_level":
                round(
                    lighting_level,
                    2
                ),

            "crowd_density":
                round(
                    crowd_density,
                    2
                ),

            "traffic_level":
                round(
                    traffic_level,
                    2
                ),

            "police_presence":
                round(
                    police_presence,
                    2
                ),

            "risk_level":
                str(
                    row.get(
                        "risk_level",
                        "Unknown"
                    )
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
    # Keep the existing mechanism:
    #
    # 70% -> severity of nearby risk
    # 30% -> density of nearby risk points
    #
    # Only the severity calculation above has been upgraded
    # to use the CSV's individual safety factors.
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
        "| average multi-factor risk =",
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
# BUILD SAFETY BREAKDOWN
# =========================================================
#
# This function only summarizes the risk_points already calculated
# by calculate_route_risk(). It does NOT calculate a second safety
# score and does NOT change the existing scoring mechanism.
# =========================================================

def build_safety_breakdown(
    risk_points
):

    if not risk_points:
        return {
            "crime_risk": None,
            "lighting_level": None,
            "crowd_density": None,
            "traffic_level": None,
            "police_presence": None,
            "existing_risk_score": None
        }


    fields = [
        "crime_risk",
        "lighting_level",
        "crowd_density",
        "traffic_level",
        "police_presence",
        "risk_score"
    ]


    breakdown = {}


    for field in fields:

        values = []

        for point in risk_points:

            try:
                value = float(
                    point.get(field)
                )
            except (
                TypeError,
                ValueError
            ):
                continue

            if math.isfinite(value):
                values.append(value)


        if values:

            breakdown[field] = round(
                sum(values) / len(values),
                2
            )

        else:

            breakdown[field] = None


    return {
        "crime_risk":
            breakdown.get("crime_risk"),

        "lighting_level":
            breakdown.get("lighting_level"),

        "crowd_density":
            breakdown.get("crowd_density"),

        "traffic_level":
            breakdown.get("traffic_level"),

        "police_presence":
            breakdown.get("police_presence"),

        "existing_risk_score":
            breakdown.get("risk_score")
    }


# =========================================================
# SAFETY RISK LEVEL
# =========================================================

def get_risk_level(safety_score):

    try:
        score = float(safety_score)
    except (TypeError, ValueError):
        return "Unknown"

    if score >= 80:
        return "Very Safe"

    elif score >= 60:
        return "Safe"

    elif score >= 40:
        return "Moderate"

    elif score >= 20:
        return "Risky"

    else:
        return "High Risk"


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
# FINAL ML + RULE-BASED RISK
# =========================================================

RULE_BASED_WEIGHT = 0.40
ML_WEIGHT = 0.60


def get_final_risk_level(score):
    """
    Convert final numerical risk into a display level.
    """

    score = float(score)

    if score >= 65:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


def calculate_final_risk(
    existing_risk,
    ml_risk,
    risk_point_count
):
    """
    Calculate the final route risk.

    Rules:
    1. No safety/risk points -> final score is None.
       This prevents an unmeasured route from appearing
       artificially safe.
    2. Safety points + ML prediction -> 40% existing risk
       + 60% ML risk.
    3. Safety points but no ML prediction -> existing risk.
    """

    existing_risk = float(existing_risk)

    if int(risk_point_count) == 0:

        return {
            "final_risk_score": None,
            "final_risk_level": "Insufficient Data",
            "risk_source": "no_safety_data"
        }

    if ml_risk is None:

        final_score = max(
            0.0,
            min(
                100.0,
                existing_risk
            )
        )

        return {
            "final_risk_score": round(
                final_score,
                2
            ),
            "final_risk_level": get_final_risk_level(
                final_score
            ),
            "risk_source": "existing_algorithm"
        }

    ml_risk = float(ml_risk)

    final_score = (
        RULE_BASED_WEIGHT * existing_risk
        +
        ML_WEIGHT * ml_risk
    )

    final_score = max(
        0.0,
        min(
            100.0,
            final_score
        )
    )

    return {
        "final_risk_score": round(
            final_score,
            2
        ),
        "final_risk_level": get_final_risk_level(
            final_score
        ),
        "risk_source": "combined_rule_ml"
    }


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


        # =====================================================
        # FIND SAFETY DATA
        # =====================================================

        nearby_safety_data = (
            get_safety_data_for_route(
                route["geometry"],
                SAFETY_RADIUS_KM
            )
        )


        # =====================================================
        # EXISTING RULE-BASED RISK
        # =====================================================
        #
        # IMPORTANT:
        # This calculation remains unchanged.
        # It is executed BEFORE ML so that ML can use the
        # exact risk_points generated for this route.
        # =====================================================

        route_risk_data = calculate_route_risk(
            route["geometry"]
        )


        calculated_risk_points = (
            route_risk_data["risk_points"]
        )


        print(
            "Route",
            index + 1,
            "| displayed safety points:",
            len(
                nearby_safety_data
            ),
            "| calculated risk points:",
            len(
                calculated_risk_points
            ),
            "| route risk:",
            route_risk_data["route_risk"],
            "| safety score:",
            route_risk_data["safety_score"]
        )


        # =====================================================
        # SAFETY BREAKDOWN
        # =====================================================

        safety_breakdown = (
            build_safety_breakdown(
                calculated_risk_points
            )
        )


        # =====================================================
        # ML RISK PREDICTION
        # =====================================================
        #
        # The Random Forest predicts numerical risk_score.
        #
        # We use the SAME calculated risk points that the
        # existing route-risk algorithm used.
        #
        # If a route has no calculated risk points, ML is
        # intentionally unavailable rather than assuming
        # zeros or artificial safety.
        # =====================================================

        ml_risk_score = None
        ml_risk_level = None

        ml_source_points = (
            calculated_risk_points
        )


        def safe_average(
            points,
            field_name
        ):

            values = []

            for point in points:

                try:

                    value = point.get(
                        field_name,
                        None
                    )

                    if value is None:
                        continue

                    value = float(value)

                    if pd.notna(value):
                        values.append(
                            value
                        )

                except (
                    TypeError,
                    ValueError
                ):
                    continue

            if not values:
                return None

            return (
                sum(values)
                /
                len(values)
            )


        crime_risk = safe_average(
            ml_source_points,
            "crime_risk"
        )

        lighting_level = safe_average(
            ml_source_points,
            "lighting_level"
        )

        crowd_density = safe_average(
            ml_source_points,
            "crowd_density"
        )

        traffic_level = safe_average(
            ml_source_points,
            "traffic_level"
        )

        police_presence = safe_average(
            ml_source_points,
            "police_presence"
        )


        ml_features_available = all([
            crime_risk is not None,
            lighting_level is not None,
            crowd_density is not None,
            traffic_level is not None,
            police_presence is not None
        ])


        if ml_features_available:

            try:

                current_time = (
                    datetime.now()
                )


                ml_result = predict_risk(

                    crime_risk=crime_risk,

                    lighting_level=lighting_level,

                    crowd_density=crowd_density,

                    traffic_level=traffic_level,

                    police_presence=police_presence,

                    hour=current_time.hour,

                    day_of_week=current_time.strftime(
                        "%A"
                    )

                )


                ml_risk_score = (
                    ml_result[
                        "predicted_risk_score"
                    ]
                )

                ml_risk_level = (
                    ml_result[
                        "predicted_risk_level"
                    ]
                )


            except Exception as error:

                print(
                    "ML prediction error:",
                    repr(error)
                )

        else:

            print(
                "ML skipped: required safety "
                "features are missing."
            )


        # =====================================================
        # ROUTE DEBUG
        # =====================================================

        print(
            "Route",
            index + 1,

            "| Existing Risk:",
            route_risk_data[
                "route_risk"
            ],

            "| Existing Safety:",
            route_risk_data[
                "safety_score"
            ],

            "| ML Risk:",
            ml_risk_score,

            "| ML Level:",
            ml_risk_level,

            "| ML Source Points:",
            len(
                ml_source_points
            ),

            "| ML Features:",
            {
                "crime": crime_risk,
                "lighting": lighting_level,
                "crowd": crowd_density,
                "traffic": traffic_level,
                "police": police_presence
            }
        )


        # =====================================================
        # FINAL RISK
        # =====================================================

        final_risk_data = calculate_final_risk(
            existing_risk=route_risk_data[
                "route_risk"
            ],
            ml_risk=ml_risk_score,
            risk_point_count=len(
                calculated_risk_points
            )
        )


        print(
            "Route",
            index + 1,
            "| Existing Risk:",
            route_risk_data[
                "route_risk"
            ],
            "| ML Risk:",
            ml_risk_score,
            "| FINAL Risk:",
            final_risk_data[
                "final_risk_score"
            ],
            "| FINAL Level:",
            final_risk_data[
                "final_risk_level"
            ]
        )


        # =====================================================
        # ROUTE RESPONSE OBJECT
        # =====================================================

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

            # Existing frontend safety data
            "safety_data":
                nearby_safety_data,

            # Existing backend risk information
            "safety_point_count":
                len(
                    calculated_risk_points
                ),

            "risk_point_count":
                len(
                    calculated_risk_points
                ),

            "route_risk":
                route_risk_data[
                    "route_risk"
                ],

            "safety_score":
                route_risk_data[
                    "safety_score"
                ],

            "risk_level":
                get_risk_level(
                    route_risk_data[
                        "safety_score"
                    ]
                ),

            # ML prediction
            "ml_risk_score":
                ml_risk_score,

            "ml_risk_level":
                ml_risk_level,

            # Final combined risk
            "final_risk_score":
                final_risk_data[
                    "final_risk_score"
                ],

            "final_risk_level":
                final_risk_data[
                    "final_risk_level"
                ],

            "risk_source":
                final_risk_data[
                    "risk_source"
                ],

            # Backend calculated risk points
            "risk_points":
                calculated_risk_points,

            # Existing breakdown
            "safety_breakdown":
                safety_breakdown

        })


    # =====================================================
    # FIND SAFEST ROUTE
    # =====================================================
    #
    # Lower final risk = safer route.
    # This now considers the Random Forest when ML data
    # is available and falls back to the existing algorithm
    # when ML data is unavailable.
    # =====================================================

    routes_with_risk = [
        route
        for route in routes
        if route.get(
            "final_risk_score"
        ) is not None
    ]

    if routes_with_risk:

        safest_route = min(
            routes_with_risk,
            key=lambda route:
                route[
                    "final_risk_score"
                ]
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

        "routes_with_risk_data":
            len(routes_with_risk),

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

        # Reload the same unified dataset used by route-risk
        # calculation and map safety-point detection.
        df = pd.read_csv(SAFETY_FILE)

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

            "error": "amravati_safety_data.csv not found",

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
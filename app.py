from flask import Flask, render_template, request, jsonify
import requests
import math
import pandas as pd
import os
from datetime import datetime

from ml_risk_predictor import predict_risk


app = Flask(__name__)


# =========================================================
# CONFIGURATION
# =========================================================

SAFETY_FILE = "amravati_safety_data.csv"
IMPORTANT_PLACES_FILE = "important_places.csv"

# Safety/risk points within this distance of a route
# are considered for route-risk calculation.
SAFETY_RADIUS_KM = 0.5

# Existing rule-based risk + ML risk weights
RULE_BASED_WEIGHT = 0.40
ML_WEIGHT = 0.60


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
            print("Safety dataset missing:", missing)
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
            f"WARNING: {SAFETY_FILE} not found"
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
            f"WARNING: {IMPORTANT_PLACES_FILE} not found"
        )
        return pd.DataFrame()

    except Exception as error:
        print(
            "Important places error:",
            error
        )
        return pd.DataFrame()


# Load data when the application starts.
safety_data = load_safety_data()
important_places = load_important_places()


# =========================================================
# UNIFIED SAFETY DATA FOR ROUTE RISK
# =========================================================

MAX_RISK_DISTANCE_KM = SAFETY_RADIUS_KM

# Use the same safety dataset for:
# 1. Map safety points
# 2. Rule-based route risk
# 3. ML feature extraction
area_risk_data = safety_data.copy()

if area_risk_data.empty:
    print("Unified safety data is empty.")

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
# NORMALIZE VALUE
# =========================================================

def normalize(value):
    """
    Convert a risk/safety factor to a 0-100 range.

    The dataset may contain values as:
        0-1
    or:
        0-100
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 0.0

    if 0 <= value <= 1:
        value *= 100

    return max(
        0.0,
        min(100.0, value)
    )


# =========================================================
# SAFE AVERAGE
# =========================================================

def safe_average(points, field_name):
    """
    Calculate the average of a numeric field from
    a list of risk points.

    Returns None when no usable values exist.
    """
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
                values.append(value)

        except (TypeError, ValueError):
            continue

    if not values:
        return None

    return sum(values) / len(values)


# =========================================================
# CALCULATE DISTANCE WEIGHT
# =========================================================

def calculate_distance_weight(distance_km):
    """
    Risk points closer to the route receive greater weight.

        0-100 m     -> 1.0
        100-300 m   -> 0.7
        300-500 m   -> 0.3
        >500 m      -> 0.0
    """
    if distance_km <= 0.1:
        return 1.0

    if distance_km <= 0.3:
        return 0.7

    if distance_km <= 0.5:
        return 0.3

    return 0.0


# =========================================================
# HAVERSINE DISTANCE
# =========================================================

def calculate_distance(
    lat1,
    lon1,
    lat2,
    lon2
):
    """
    Calculate distance between two latitude/longitude
    coordinates in kilometres.
    """
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
# CALCULATE ROUTE RISK
# =========================================================

def calculate_route_risk(route_geometry):
    """
    Existing rule-based route-risk mechanism.

    Point-level risk uses:
        Crime Risk       = 30%
        Lighting Risk    = 20%
        Crowd Density    = 15%
        Traffic Level    = 10%
        Police Risk      = 15%
        Existing Risk    = 10%

    Route-level risk uses:
        Severity         = 70%
        Risk Density     = 30%
    """

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

        except (TypeError, ValueError):
            continue

        # Find the nearest route coordinate
        minimum_distance = float("inf")

        for coordinate in coordinates:

            if not coordinate or len(coordinate) < 2:
                continue

            try:
                route_lon = float(
                    coordinate[0]
                )

                route_lat = float(
                    coordinate[1]
                )

            except (TypeError, ValueError):
                continue

            distance = calculate_distance(
                route_lat,
                route_lon,
                risk_lat,
                risk_lon
            )

            if distance < minimum_distance:
                minimum_distance = distance

        # Ignore locations farther than 500 metres.
        if minimum_distance > MAX_RISK_DISTANCE_KM:
            continue

        # =================================================
        # READ MULTI-FACTOR DATA
        # =================================================

        crime_risk = normalize(
            row.get("crime_risk", 0)
        )

        lighting_level = normalize(
            row.get("lighting_level", 0)
        )

        crowd_density = normalize(
            row.get("crowd_density", 0)
        )

        traffic_level = normalize(
            row.get("traffic_level", 0)
        )

        police_presence = normalize(
            row.get("police_presence", 0)
        )

        existing_risk = normalize(
            row.get("risk_score", 0)
        )

        # Higher lighting = safer.
        lighting_risk = 100.0 - lighting_level

        # Higher police presence = safer.
        police_risk = 100.0 - police_presence

        # Crowd density is treated as a risk factor
        # in the current prototype.
        crowd_risk = crowd_density

        # =================================================
        # MULTI-FACTOR POINT RISK
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
            min(100.0, point_risk)
        )

        # =================================================
        # DISTANCE WEIGHTING
        # =================================================

        weight = calculate_distance_weight(
            minimum_distance
        )

        weighted_risk = (
            point_risk * weight
        )

        # =================================================
        # STORE RISK POINT
        # =================================================

        risk_points.append({
            "location_name": str(
                row.get(
                    "location_name",
                    row.get(
                        "police_station",
                        "Risk area"
                    )
                )
            ),

            "latitude": risk_lat,
            "longitude": risk_lon,

            "distance_km": round(
                minimum_distance,
                3
            ),

            # Existing dataset risk score
            "risk_score": round(
                existing_risk,
                2
            ),

            # New calculated multi-factor risk
            "calculated_point_risk": round(
                point_risk,
                2
            ),

            "weight": weight,

            "weighted_risk": round(
                weighted_risk,
                2
            ),

            "crime_risk": round(
                crime_risk,
                2
            ),

            "lighting_level": round(
                lighting_level,
                2
            ),

            "crowd_density": round(
                crowd_density,
                2
            ),

            "traffic_level": round(
                traffic_level,
                2
            ),

            "police_presence": round(
                police_presence,
                2
            ),

            "risk_level": str(
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
            "route_risk": 0,
            "safety_score": 100,
            "risk_points": []
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
            / total_weight
        )
    else:
        average_risk = 0

    average_risk = max(
        0.0,
        min(100.0, average_risk)
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
    # FINAL RULE-BASED ROUTE RISK
    # =====================================================

    route_risk = (
        average_risk * 0.70
        +
        density_risk * 0.30
    )

    route_risk = max(
        0.0,
        min(100.0, route_risk)
    )

    # Higher safety score = safer route.
    safety_score = 100 - route_risk

    safety_score = max(
        0.0,
        min(100.0, safety_score)
    )

    print(
        "Route risk calculation:",
        "points =", risk_point_count,
        "| average multi-factor risk =",
        round(average_risk, 2),
        "| density risk =",
        round(density_risk, 2),
        "| route risk =",
        round(route_risk, 2),
        "| safety score =",
        round(safety_score, 2)
    )

    return {
        "route_risk": round(
            route_risk,
            2
        ),

        "safety_score": round(
            safety_score,
            2
        ),

        "risk_points": risk_points
    }


# =========================================================
# BUILD SAFETY BREAKDOWN
# =========================================================

def build_safety_breakdown(risk_points):
    """
    Summarize the calculated risk points.

    This does not calculate another route score.
    """

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
            except (TypeError, ValueError):
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
        score = float(
            safety_score
        )
    except (TypeError, ValueError):
        return "Unknown"

    if score >= 80:
        return "Very Safe"

    if score >= 60:
        return "Safe"

    if score >= 40:
        return "Moderate"

    if score >= 20:
        return "Risky"

    return "High Risk"


# =========================================================
# FINAL ML + RULE-BASED RISK
# =========================================================

def get_final_risk_level(score):
    """
    Convert final numerical risk into a display level.

    Lower risk score = safer route.
    """

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Unknown"

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
    Combine the existing route-risk algorithm with ML.

    Rules:
        1. No safety points:
           final risk = None

        2. Safety points + ML:
           40% existing risk + 60% ML risk

        3. Safety points but ML unavailable:
           use existing risk
    """

    try:
        existing_risk = float(
            existing_risk
        )
    except (TypeError, ValueError):
        existing_risk = 0.0

    existing_risk = max(
        0.0,
        min(100.0, existing_risk)
    )

    if int(risk_point_count) == 0:
        return {
            "final_risk_score": None,
            "final_risk_level": "Insufficient Data",
            "risk_source": "no_safety_data"
        }

    if ml_risk is None:
        final_score = existing_risk

        return {
            "final_risk_score": round(
                final_score,
                2
            ),
            "final_risk_level":
                get_final_risk_level(
                    final_score
                ),
            "risk_source":
                "existing_algorithm"
        }

    try:
        ml_risk = float(
            ml_risk
        )
    except (TypeError, ValueError):
        final_score = existing_risk

        return {
            "final_risk_score": round(
                final_score,
                2
            ),
            "final_risk_level":
                get_final_risk_level(
                    final_score
                ),
            "risk_source":
                "existing_algorithm"
        }

    ml_risk = max(
        0.0,
        min(100.0, ml_risk)
    )

    final_score = (
        RULE_BASED_WEIGHT * existing_risk
        +
        ML_WEIGHT * ml_risk
    )

    final_score = max(
        0.0,
        min(100.0, final_score)
    )

    return {
        "final_risk_score": round(
            final_score,
            2
        ),

        "final_risk_level":
            get_final_risk_level(
                final_score
            ),

        "risk_source":
            "combined_rule_ml"
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
            try:
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

            except (KeyError, TypeError, ValueError):
                continue

        return locations

    except Exception as error:
        print(
            "Nominatim error:",
            error
        )
        return []


# =========================================================
# SEARCH LOCATION API
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

    # First search the local dataset.
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

    # Fallback to Nominatim.
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
# GENERATE ALTERNATIVE WAYPOINTS
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
            perpendicular_lat * deviation,

            middle_lon
            +
            perpendicular_lon * deviation
        ),

        (
            middle_lat
            -
            perpendicular_lat * deviation,

            middle_lon
            -
            perpendicular_lon * deviation
        ),

        (
            middle_lat
            +
            perpendicular_lat * deviation * 2,

            middle_lon
            +
            perpendicular_lon * deviation * 2
        ),

        (
            middle_lat
            -
            perpendicular_lat * deviation * 2,

            middle_lon
            -
            perpendicular_lon * deviation * 2
        )
    ]


# =========================================================
# OSRM ROUTE
# =========================================================

def get_osrm_route(coordinates):
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

        if not point or len(point) < 2:
            continue

        try:
            route_lon = float(point[0])
            route_lat = float(point[1])
        except (TypeError, ValueError):
            continue

        for _, row in safety_data.iterrows():

            try:
                data_lat = float(
                    row["latitude"]
                )

                data_lon = float(
                    row["longitude"]
                )

            except (TypeError, ValueError):
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
                        value = float(value)

                    else:
                        value = str(value)

                    safety_point[column] = value

                nearby_points.append(
                    safety_point
                )

    # Remove duplicate points.
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
            unique_points.append(point)

    return unique_points


# =========================================================
# ROUTE API
# =========================================================

@app.route(
    "/api/route",
    methods=["POST"]
)
def get_multiple_routes():

    data = request.get_json(
        silent=True
    )

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
        start_lat = float(start_lat)
        start_lon = float(start_lon)
        end_lat = float(end_lat)
        end_lon = float(end_lon)

    except (TypeError, ValueError):
        return jsonify({
            "success": False,
            "message":
                "Coordinates must be numbers"
        }), 400

    # Same location check.
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

        candidate_routes.append(
            route
        )

    # Sort by distance.
    candidate_routes.sort(
        key=lambda route:
            route["distance"]
    )

    # Maximum four routes.
    candidate_routes = candidate_routes[:4]

    print(
        "\n================ ROUTE DEBUG ================"
    )

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

    print(
        "=============================================\n"
    )

    routes = []

    # =====================================================
    # FORMAT AND SCORE ROUTES
    # =====================================================

    for index, route in enumerate(
        candidate_routes
    ):

        distance_km = (
            route["distance"] / 1000
        )

        duration_minutes = (
            route["duration"] / 60
        )

        # =================================================
        # FIND SAFETY DATA
        # =================================================

        nearby_safety_data = (
            get_safety_data_for_route(
                route["geometry"],
                SAFETY_RADIUS_KM
            )
        )

        # =================================================
        # EXISTING RULE-BASED RISK
        # =================================================

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
            len(nearby_safety_data),
            "| calculated risk points:",
            len(calculated_risk_points),
            "| route risk:",
            route_risk_data["route_risk"],
            "| safety score:",
            route_risk_data["safety_score"]
        )

        # =================================================
        # SAFETY BREAKDOWN
        # =================================================

        safety_breakdown = (
            build_safety_breakdown(
                calculated_risk_points
            )
        )

        # =================================================
        # ML RISK PREDICTION
        # =================================================

        ml_risk_score = None
        ml_risk_level = None

        ml_source_points = (
            calculated_risk_points
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
                current_time = datetime.now()

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
                    ml_result.get(
                        "predicted_risk_score"
                    )
                )

                ml_risk_level = (
                    ml_result.get(
                        "predicted_risk_level"
                    )
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

        # =================================================
        # ROUTE DEBUG
        # =================================================

        print(
            "Route",
            index + 1,
            "| Existing Risk:",
            route_risk_data["route_risk"],
            "| Existing Safety:",
            route_risk_data["safety_score"],
            "| ML Risk:",
            ml_risk_score,
            "| ML Level:",
            ml_risk_level,
            "| ML Source Points:",
            len(ml_source_points),
            "| ML Features:",
            {
                "crime": crime_risk,
                "lighting": lighting_level,
                "crowd": crowd_density,
                "traffic": traffic_level,
                "police": police_presence
            }
        )

        # =================================================
        # FINAL COMBINED RISK
        # =================================================

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
            route_risk_data["route_risk"],
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

        # =================================================
        # ROUTE RESPONSE OBJECT
        # =================================================

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

            # Frontend safety data
            "safety_data":
                nearby_safety_data,

            # Backend risk information
            "safety_point_count":
                len(calculated_risk_points),

            "risk_point_count":
                len(calculated_risk_points),

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

            # Detailed calculated points
            "risk_points":
                calculated_risk_points,

            # Safety factor averages
            "safety_breakdown":
                safety_breakdown
        })

    # =====================================================
    # FIND SAFEST ROUTE
    # =====================================================

    # Lower final risk = safer route.
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
                route["final_risk_score"]
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

        "safest_route_id":
            safest_route_id,

        # Return all generated routes.
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

@app.route(
    "/api/safety-data",
    methods=["GET"]
)
def get_safety_data():

    try:
        # Reload the same dataset used by
        # route-risk calculation and the map.
        df = pd.read_csv(
            SAFETY_FILE
        )

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

        df = df.dropna(
            subset=[
                "latitude",
                "longitude"
            ]
        )

        # Convert NaN values to None for valid JSON.
        df = df.where(
            pd.notnull(df),
            None
        )

        safety_points = df.to_dict(
            orient="records"
        )

        print(
            f"Sending {len(safety_points)} "
            "safety points to frontend"
        )

        return jsonify({
            "success": True,
            "count":
                len(safety_points),
            "data":
                safety_points
        })

    except FileNotFoundError:
        return jsonify({
            "success": False,
            "error":
                f"{SAFETY_FILE} not found",
            "data": []
        }), 404

    except Exception as error:
        print(
            "Safety data error:",
            error
        )

        return jsonify({
            "success": False,
            "error":
                str(error),
            "data": []
        }), 500


# =========================================================
# HEALTH CHECK
# =========================================================

@app.route(
    "/api/health",
    methods=["GET"]
)
def health_check():
    return jsonify({
        "success": True,
        "message":
            "Women Safety GPS API is running",
        "safety_records":
            len(safety_data),
        "important_places":
            len(important_places)
    })


# =========================================================
# RUN
# =========================================================

if __name__ == "__main__":

    # Works locally and also respects Render's PORT.
    port = int(
        os.environ.get(
            "PORT",
            5000
        )
    )

    debug = (
        os.environ.get(
            "FLASK_DEBUG",
            "false"
        ).lower()
        == "true"
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=debug
    )

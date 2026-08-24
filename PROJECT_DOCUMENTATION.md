# Smart Safe Route Recommendation System

## 1. Project Overview

The Smart Safe Route Recommendation System is a safety-aware route planning application that generates multiple candidate routes between a source and destination and evaluates each route using geographic safety data and a Machine Learning model.

The system combines:

1. Geographic route generation
2. Safety-point detection
3. Existing multi-factor risk calculation
4. Random Forest risk prediction
5. Final risk-score calculation
6. Route ranking
7. Safest-route recommendation

The objective is not simply to find the shortest route, but to identify a route with a lower estimated safety risk while considering travel distance and duration.

---

# 2. System Workflow

```text
User
  |
  v
Enter Start Location
  |
  v
Enter Destination
  |
  v
Generate Candidate Routes
  |
  v
Find Safety Points Near Each Route
  |
  v
Calculate Existing Route Risk
  |
  v
Extract ML Features
  |
  v
Random Forest Prediction
  |
  v
Combine Existing Risk + ML Risk
  |
  v
Calculate Final Risk
  |
  v
Classify Risk Level
  |
  v
Rank Candidate Routes
  |
  v
Recommend Safest Route
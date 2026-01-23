"""
Store Locator Module - Fetches nearby grocery stores using OpenStreetMap Overpass API.
"""
import requests
import time
from typing import List, Dict, Optional
import math


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two lat/lon points in kilometers using Haversine formula."""
    R = 6371  # Earth radius in km
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))
    
    return R * c


def geocode_location(location: Dict[str, Optional[str]], timeout: int = 10) -> Optional[tuple]:
    """
    Geocode a location to lat/lon using Nominatim (OpenStreetMap).
    
    Args:
        location: Dict with keys: country, city, state, zip (all optional)
        timeout: Request timeout in seconds
    
    Returns:
        (lat, lon) tuple or None if geocoding fails
    """
    try:
        # Build query string - prioritize zip code if available for more accurate results
        query_parts = []
        if location.get("zip"):
            query_parts.append(location["zip"])
        if location.get("city"):
            query_parts.append(location["city"])
        if location.get("state"):
            query_parts.append(location["state"])
        if location.get("country"):
            query_parts.append(location["country"])
        
        if not query_parts:
            return None
        
        query = ", ".join(query_parts)
        
        # Call Nominatim API
        url = "https://nominatim.openstreetmap.org/search"
        params = {
            "q": query,
            "format": "json",
            "limit": 1
        }
        headers = {
            "User-Agent": "FinanceAssistant/1.0"  # Required by Nominatim
        }
        
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        if data and len(data) > 0:
            lat = float(data[0]["lat"])
            lon = float(data[0]["lon"])
            return (lat, lon)
        
        return None
    except Exception as e:
        print(f"Geocoding error: {e}")
        return None


def get_nearby_grocery_stores(location: Dict[str, Optional[str]], radius_km: float = 10) -> List[Dict]:
    """
    Fetch nearby grocery stores using OpenStreetMap Overpass API.
    
    Args:
        location: Dict with keys: country, city, state (all optional)
        radius_km: Search radius in kilometers (default 10km)
    
    Returns:
        List of store dicts with keys: name, address, city, state, country, distance_km, source
    """
    # Geocode location first
    coords = geocode_location(location)
    if not coords:
        return []
    
    lat, lon = coords
    
    try:
        # Overpass API query for nearby supermarkets and grocery stores
        # Search for: supermarket, grocery_store, convenience (filtered to grocery/supermarket)
        overpass_url = "https://overpass-api.de/api/interpreter"
        
        # Convert radius to meters for Overpass
        radius_m = int(radius_km * 1000)
        
        # Overpass QL query
        query = f"""
        [out:json][timeout:25];
        (
          node["shop"="supermarket"](around:{radius_m},{lat},{lon});
          node["shop"="grocery"](around:{radius_m},{lat},{lon});
          way["shop"="supermarket"](around:{radius_m},{lat},{lon});
          way["shop"="grocery"](around:{radius_m},{lat},{lon});
        );
        out body;
        >;
        out skel qt;
        """
        
        response = requests.post(overpass_url, data=query, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        
        stores = []
        seen = set()  # For deduplication
        
        if "elements" not in data:
            return []
        
        for element in data["elements"]:
            if "tags" not in element:
                continue
            
            tags = element["tags"]
            name = tags.get("name", "").strip()
            
            if not name:
                continue
            
            # Get coordinates
            if element["type"] == "node":
                store_lat = element.get("lat")
                store_lon = element.get("lon")
            elif element["type"] == "way" and "center" in element:
                store_lat = element["center"].get("lat")
                store_lon = element["center"].get("lon")
            else:
                continue
            
            if not store_lat or not store_lon:
                continue
            
            # Calculate distance
            distance = haversine_distance(lat, lon, store_lat, store_lon)
            
            # Build address
            address_parts = []
            if tags.get("addr:housenumber") and tags.get("addr:street"):
                address_parts.append(f"{tags['addr:housenumber']} {tags['addr:street']}")
            elif tags.get("addr:street"):
                address_parts.append(tags["addr:street"])
            
            city = tags.get("addr:city") or tags.get("addr:place") or location.get("city", "")
            state = tags.get("addr:state") or location.get("state", "")
            country = tags.get("addr:country") or location.get("country", "")
            
            if city:
                address_parts.append(city)
            if state:
                address_parts.append(state)
            if country:
                address_parts.append(country)
            
            address = ", ".join(address_parts) if address_parts else "Address not available"
            
            # Deduplicate by name + address
            dedup_key = (name.lower(), address.lower())
            if dedup_key in seen:
                continue
            seen.add(dedup_key)
            
            stores.append({
                "name": name,
                "address": address,
                "city": city,
                "state": state,
                "country": country,
                "distance_km": round(distance, 2),
                "source": "OpenStreetMap"
            })
        
        # Sort by distance (closest first)
        stores.sort(key=lambda x: x["distance_km"] if x["distance_km"] is not None else float('inf'))
        
        # Limit to top 10
        return stores[:10]
        
    except requests.exceptions.Timeout:
        print("Overpass API timeout")
        return []
    except requests.exceptions.RequestException as e:
        print(f"Overpass API error: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error in store lookup: {e}")
        return []


def filter_stores_by_name(stores: List[Dict], store_name: str) -> List[Dict]:
    """
    Filter stores by name (case-insensitive partial match).
    
    Args:
        stores: List of store dicts
        store_name: Store name to search for
    
    Returns:
        Filtered list of stores
    """
    if not store_name:
        return stores
    
    store_name_lower = store_name.lower().strip()
    filtered = []
    
    for store in stores:
        if store_name_lower in store["name"].lower():
            filtered.append(store)
    
    return filtered

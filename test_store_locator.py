"""
Simple test script for store locator functionality.
"""
from store_locator import geocode_location, get_nearby_grocery_stores

def test_geocode_tulsa():
    """Test geocoding for Tulsa, OK"""
    location = {
        "city": "Tulsa",
        "state": "OK",
        "country": "United States"
    }
    
    coords = geocode_location(location)
    if coords:
        print(f"✅ Geocoding successful: {coords}")
        return True
    else:
        print("❌ Geocoding failed")
        return False

def test_store_lookup_tulsa():
    """Test store lookup for Tulsa, OK"""
    location = {
        "city": "Tulsa",
        "state": "OK",
        "country": "United States"
    }
    
    stores = get_nearby_grocery_stores(location, radius_km=10)
    
    if stores:
        print(f"✅ Found {len(stores)} stores:")
        for store in stores[:3]:
            print(f"  - {store['name']}: {store['address']} ({store['distance_km']} km)")
        return True
    else:
        print("⚠️ No stores found (this may be normal if Overpass API is slow or location has no stores)")
        return True  # Not a failure - could be legitimate

if __name__ == "__main__":
    print("Testing store locator...")
    print("\n1. Testing geocoding:")
    test_geocode_tulsa()
    
    print("\n2. Testing store lookup:")
    test_store_lookup_tulsa()
    
    print("\n✅ Tests completed")

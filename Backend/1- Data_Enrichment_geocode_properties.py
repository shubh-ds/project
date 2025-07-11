import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from tqdm import tqdm
import logging
import os
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('geocoding_log.txt'),
        logging.StreamHandler()
    ]
)

def create_robust_geocoder():
    """Create a new geocoder instance with proper timeout"""
    return Nominatim(
        user_agent="mumbai_real_estate_research_v1",
        timeout=30  # 30 second timeout
    )

def create_address_variants(row):
    """Create multiple address variants to try for geocoding"""
    addresses = []
    
    # Clean and prepare components
    landmark = str(row['Landmark']).strip() if pd.notna(row['Landmark']) else ''
    project_name = str(row['Project Name']).strip() if pd.notna(row['Project Name']) else ''
    area_name = str(row['Area Name']).strip() if pd.notna(row['Area Name']) else ''
    city = str(row['City']).strip() if pd.notna(row['City']) else ''
    
    # Multiple variants from most specific to least specific
    
    # Variant 1: Project Name + Area + City
    if project_name and project_name != 'nan':
        addresses.append(f"{project_name}, {area_name}, {city}, Maharashtra, India")
    
    # Variant 2: Area + City (often more reliable than project names)
    if area_name and city:
        addresses.append(f"{area_name}, {city}, Maharashtra, India")
    
    # Variant 3: Area + Mumbai (sometimes works better)
    if area_name:
        addresses.append(f"{area_name}, Mumbai, Maharashtra, India")
    
    # Variant 4: Landmark + Area if available
    if landmark and landmark != 'nan' and area_name:
        addresses.append(f"{landmark}, {area_name}, Mumbai, Maharashtra, India")
    
    # Variant 5: Just city as last resort
    if city:
        addresses.append(f"{city}, Maharashtra, India")
    
    return addresses

def geocode_with_retry(geolocator, address, max_retries=3):
    """Geocode an address with retry logic and timeout handling"""
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=30)
            if location:
                return {
                    'latitude': location.latitude,
                    'longitude': location.longitude,
                    'geocoded_address': location.address,
                    'success': True
                }
            return None
            
        except GeocoderTimedOut:
            wait_time = 5 * (attempt + 1)
            logging.warning(f"Timeout for {address}, attempt {attempt + 1}. Waiting {wait_time}s...")
            time.sleep(wait_time)
            if attempt == max_retries - 1:
                return None
                
        except Exception as e:
            logging.error(f"Error geocoding {address}: {str(e)}")
            time.sleep(5)
            return None

def load_progress():
    """Check for existing progress and load it"""
    if os.path.exists('Backend\\Output Files\\properties_with_coordinates_partial.csv'):
        logging.info("Found existing progress file. Loading...")
        return pd.read_csv('Backend\\Output Files\\properties_with_coordinates_partial.csv')
    return None

def main():
    # Load the dataset
    logging.info("Loading dataset...")
    df = pd.read_csv('Data\\properties.csv')
    
    # Drop properties with NA Property name
    mask = pd.notna(df['Project Name'])
    df = df[mask].reset_index(drop=True)
    logging.info(f"Filtered dataset: {len(df)} properties with project names")
    
    # Check for existing progress
    df_existing = load_progress()
    if df_existing is not None:
        # Find already geocoded properties
        already_geocoded = set(df_existing[df_existing['geocoding_success']]['ID'].tolist())
        logging.info(f"Found {len(already_geocoded)} already geocoded properties")
    else:
        already_geocoded = set()
    
    # Initialize result columns if not resuming
    if 'latitude' not in df.columns:
        df['latitude'] = np.nan
        df['longitude'] = np.nan
        df['geocoded_address'] = ''
        df['geocoding_success'] = False
        df['address_variant_used'] = ''
    
    # Initialize counters and geocoder
    geolocator = create_robust_geocoder()
    successful_geocodes = len(already_geocoded)
    failed_geocodes = 0
    requests_since_refresh = 0
    consecutive_failures = 0
    start_time = datetime.now()
    
    # Main geocoding loop
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Geocoding properties"):
        # Skip if already geocoded
        if row['ID'] in already_geocoded:
            continue
        
        requests_since_refresh += 1
        
        # Refresh geocoder connection every 300 requests
        if requests_since_refresh >= 300:
            logging.info("Refreshing geocoder connection...")
            geolocator = create_robust_geocoder()
            requests_since_refresh = 0
            time.sleep(60)  # 1 minute break
        
        # Take longer breaks periodically
        if successful_geocodes > 0 and successful_geocodes % 500 == 0:
            logging.info(f"Taking 5-minute break after {successful_geocodes} successful geocodes...")
            time.sleep(300)
        
        # Get address variants
        addresses = create_address_variants(row)
        
        if not addresses:
            logging.warning(f"No valid addresses for property ID {row['ID']}")
            failed_geocodes += 1
            continue
        
        # Try each address variant
        geocoded = False
        for i, address in enumerate(addresses):
            result = geocode_with_retry(geolocator, address)
            
            if result:
                df.at[idx, 'latitude'] = result['latitude']
                df.at[idx, 'longitude'] = result['longitude']
                df.at[idx, 'geocoded_address'] = result['geocoded_address']
                df.at[idx, 'geocoding_success'] = True
                df.at[idx, 'address_variant_used'] = f"Variant_{i+1}: {address}"
                successful_geocodes += 1
                consecutive_failures = 0
                geocoded = True
                break
            
            # Rate limiting between address attempts
            time.sleep(2)
        
        if not geocoded:
            failed_geocodes += 1
            consecutive_failures += 1
            logging.warning(f"Failed to geocode property ID {row['ID']}")
            
            # Exponential backoff on consecutive failures
            if consecutive_failures > 5:
                wait_time = min(300, 10 * consecutive_failures)
                logging.warning(f"{consecutive_failures} consecutive failures. Waiting {wait_time}s...")
                time.sleep(wait_time)
                # Refresh connection after many failures
                geolocator = create_robust_geocoder()
        
        # Save progress every 100 properties
        if (successful_geocodes + failed_geocodes) % 100 == 0:
            df.to_csv('Backend\\Output Files\\properties_with_coordinates_partial.csv', index=False)
            elapsed = (datetime.now() - start_time).total_seconds() / 60
            rate = (successful_geocodes + failed_geocodes) / elapsed if elapsed > 0 else 0
            logging.info(f"Progress saved. Rate: {rate:.1f} properties/min")
        
        # Rate limiting between properties
        time.sleep(2)
    
    # Final summary
    total_time = (datetime.now() - start_time).total_seconds() / 60
    logging.info(f"\nGeocoding completed in {total_time:.1f} minutes")
    logging.info(f"Total properties: {len(df)}")
    logging.info(f"Successfully geocoded: {successful_geocodes} ({successful_geocodes/len(df)*100:.1f}%)")
    logging.info(f"Failed to geocode: {failed_geocodes} ({failed_geocodes/len(df)*100:.1f}%)")
    
    # Save final results
    output_file = 'Backend\\Output Files\\properties_with_coordinates.csv'
    df.to_csv(output_file, index=False)
    logging.info(f"Results saved to: {output_file}")
    
    # Create area-level fallback coordinates for failed geocodes
    if successful_geocodes > 0:
        area_coords = df[df['geocoding_success']].groupby('Area Name')[['latitude', 'longitude']].mean()
        area_coords.to_csv('Backend\\Output Files\\area_coordinates_fallback.csv')
        logging.info(f"Created fallback coordinates for {len(area_coords)} areas")
        
        # Fill failed geocodes with area averages
        filled_count = 0
        for idx, row in df[~df['geocoding_success']].iterrows():
            if row['Area Name'] in area_coords.index:
                df.at[idx, 'latitude'] = area_coords.loc[row['Area Name'], 'latitude']
                df.at[idx, 'longitude'] = area_coords.loc[row['Area Name'], 'longitude']
                df.at[idx, 'geocoded_address'] = f"Area Average: {row['Area Name']}"
                filled_count += 1
        
        if filled_count > 0:
            logging.info(f"Filled {filled_count} failed properties with area averages")
            df.to_csv('Backend\\Output Files\\properties_with_coordinates_final.csv', index=False)
            logging.info("Final results with area fallbacks saved to: Backend\\Output Files\\properties_with_coordinates_final.csv")
    
    # Clean up partial file
    if os.path.exists('Backend\\Output Files\\properties_with_coordinates_partial.csv'):
        os.remove('Backend\\Output Files\\properties_with_coordinates_partial.csv')
    
    logging.info("Geocoding process complete!")

if __name__ == "__main__":
    main()
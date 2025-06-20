from typing import Dict, Any, Optional, Callable, Union
from requests import Session
from functools import wraps
import pandas as pd

PAGINATED_ENDPOINTS = {
    'catalogs': True,                
    'datasets': True,                
    'lineage': True,                 
    'products': True,                
    'resources': True,               
    'jobs': True,                    
    'runs': True,                    
    # Add more as you discover paginated endpoints in your project
}

def requires_pagination(url: str) -> bool:
    """Check if the endpoint requires pagination"""
    return any(endpoint in url for endpoint, enabled in PAGINATED_ENDPOINTS.items() if enabled)


def handle_api_request(session: Session, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Handle API requests with optional pagination support"""
    if requires_pagination(url):
        return _handle_paginated_request(session, url, headers)
    return _handle_simple_request(session, url, headers)

def _handle_simple_request(session: Session, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Handle non-paginated GET request"""
    response = session.get(url, headers=headers or {})
    response.raise_for_status()
    return response.json()

def _handle_paginated_request(session: Session, url: str, headers: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Handle paginated GET request"""
    all_responses = []
    current_headers = headers or {}
    next_token = None
    
    while True:
        if next_token:
            current_headers['x-jpmc-next-token'] = next_token
            
        response = session.get(url, headers=current_headers)
        response.raise_for_status()
        
        response_data = response.json()
        all_responses.append(response_data)
        
        next_token = response.headers.get('x-jpmc-next-token')
        if not next_token:
            break
    
    return _merge_responses(all_responses)

def _merge_responses(responses: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Merges multiple response dictionaries, focusing on 'resources' key."""
    if not responses:
        return {"resources": []}
    merged = responses[0].copy()
    if "resources" in merged:
        merged["resources"] = list(merged["resources"])  # ensure it's a list
        for response in responses[1:]:
            if "resources" in response:
                merged["resources"].extend(response["resources"])
    return merged
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from typing import Optional
import traceback
from neo4j import GraphDatabase
from app.config import get_settings
from app.schemas.config import (
    UserConfig, 
    ConfigRequest, 
    ConfigResponse, 
    ConnectionTestRequest, 
    ConnectionTestResponse
)
from app.services.config_service import config_service
from app.utils.logger import logger

settings = get_settings()
router = APIRouter(prefix=settings.API_V1_STR, tags=["Configuration"])


@router.get("/config", response_model=UserConfig)
async def get_user_config(
    email: str = Query(..., description="User's email address")
) -> UserConfig:
    """
    Get user configuration by email
    
    Args:
        email: User's email address
        
    Returns:
        UserConfig: User's database configuration
        
    Raises:
        HTTPException: 404 if configuration not found
    """
    try:
        user_config = await config_service.get_user_config(email)
        
        if not user_config:
            raise HTTPException(
                status_code=404,
                detail=f"Configuration not found for user: {email}"
            )
        
        return user_config
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving configuration for {email}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/config", response_model=UserConfig)
async def create_user_config(config_request: ConfigRequest) -> UserConfig:
    """
    Create a new user configuration
    
    Args:
        config_request: Configuration data
        
    Returns:
        UserConfig: Created configuration
        
    Raises:
        HTTPException: 400 if validation fails or user already exists
    """
    try:
        # Validate that staging and prod databases are different
        if config_request.stagingDb.uri == config_request.prodDb.uri:
            raise HTTPException(
                status_code=400,
                detail="Staging and production database URIs must be different"
            )
        
        user_config = await config_service.create_user_config(config_request)
        return user_config
        
    except ValueError as e:
        logger.warning(f"Validation error creating config for {config_request.userEmail}: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error creating configuration for {config_request.userEmail}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.put("/config", response_model=UserConfig)
async def update_user_config(config_request: ConfigRequest) -> UserConfig:
    """
    Update an existing user configuration
    
    Args:
        config_request: Updated configuration data
        
    Returns:
        UserConfig: Updated configuration
        
    Raises:
        HTTPException: 400 if validation fails, 404 if user not found
    """
    try:
        # Validate that staging and prod databases are different
        if config_request.stagingDb.uri == config_request.prodDb.uri:
            raise HTTPException(
                status_code=400,
                detail="Staging and production database URIs must be different"
            )
        
        user_config = await config_service.update_user_config(config_request)
        return user_config
        
    except ValueError as e:
        logger.warning(f"Validation error updating config for {config_request.userEmail}: {str(e)}")
        raise HTTPException(
            status_code=404,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Error updating configuration for {config_request.userEmail}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.delete("/config")
async def delete_user_config(
    email: str = Query(..., description="User's email address")
) -> JSONResponse:
    """
    Delete a user configuration
    
    Args:
        email: User's email address
        
    Returns:
        JSONResponse: Success message
    """
    try:
        await config_service.delete_user_config(email)
        return JSONResponse(
            content={"message": f"Configuration deleted for user: {email}"},
            status_code=200
        )
        
    except Exception as e:
        logger.error(f"Error deleting configuration for {email}: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.post("/config/test-connection", response_model=ConnectionTestResponse)
async def test_neo4j_connection(request: ConnectionTestRequest) -> ConnectionTestResponse:
    """
    Test Neo4j database connection
    
    Args:
        request: Connection parameters
        
    Returns:
        ConnectionTestResponse: Test result
    """
    try:
        # Validate URI format
        valid_prefixes = ['neo4j://', 'bolt://', 'neo4j+s://', 'bolt+s://']
        if not any(request.uri.startswith(prefix) for prefix in valid_prefixes):
            return ConnectionTestResponse(
                success=False,
                message="Invalid Neo4j URI format",
                error="URI must start with neo4j://, bolt://, neo4j+s://, or bolt+s://"
            )
        
        # Test connection
        driver = GraphDatabase.driver(
            request.uri,
            auth=(request.username, request.password),
            connection_timeout=10,  # 10 seconds
            max_connection_lifetime=30,  # 30 seconds
        )
        
        try:
            # Verify connectivity
            with driver.session() as session:
                result = session.run("RETURN 1 as test")
                record = result.single()
                
                if record and record["test"] == 1:
                    return ConnectionTestResponse(
                        success=True,
                        message="Connection successful! Neo4j database is reachable and responding correctly."
                    )
                else:
                    return ConnectionTestResponse(
                        success=False,
                        message="Connection test failed",
                        error="Unexpected response from database"
                    )
        finally:
            driver.close()
            
    except Exception as e:
        error_message = str(e)
        logger.error(f"Neo4j connection test failed: {error_message}")
        
        # Provide specific error messages
        if "authentication" in error_message.lower() or "credentials" in error_message.lower():
            message = "Authentication failed"
            error = "Invalid username or password"
        elif "connection" in error_message.lower() or "connect" in error_message.lower():
            message = "Connection failed"
            error = "Unable to connect to Neo4j database. Check URI and network connectivity."
        elif "timeout" in error_message.lower():
            message = "Connection timeout"
            error = "Database did not respond within the timeout period"
        elif "ENOTFOUND" in error_message or "ECONNREFUSED" in error_message:
            message = "Database unreachable"
            error = "Cannot reach the database server. Check the URI and ensure Neo4j is running."
        elif "ServiceUnavailable" in error_message:
            message = "Service unavailable"
            error = "Neo4j service is not available. Check if the database is running and accessible."
        elif "Neo.ClientError.Security.Unauthorized" in error_message:
            message = "Authentication failed"
            error = "Invalid username or password"
        else:
            message = "Connection failed"
            error = error_message
        
        return ConnectionTestResponse(
            success=False,
            message=message,
            error=error
        ) 
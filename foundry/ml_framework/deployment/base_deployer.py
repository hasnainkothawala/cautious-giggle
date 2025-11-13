"""Base deployer abstract class.

Defines the interface for model deployment and serving.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from pathlib import Path

from ml_framework.models.base_model import BaseModel


class BaseDeployer(ABC):
    """Base class for model deployment strategies."""
    
    @abstractmethod
    def deploy(
        self,
        model: BaseModel,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Deploy model.
        
        Args:
            model: Model to deploy
            config: Deployment config
        
        Returns:
            Deployment info dict
        """
        pass
    
    @abstractmethod
    def undeploy(self, deployment_id: str) -> None:
        """Remove a deployment.
        
        Args:
            deployment_id: Deployment ID
        """
        pass
    
    @abstractmethod
    def list_deployments(self) -> list[Dict[str, Any]]:
        """List active deployments.
        
        Returns:
            List of deployment info dicts
        """
        pass


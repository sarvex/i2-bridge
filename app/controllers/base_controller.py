from typing import Generic, TypeVar
from app.repositories.base_repository import BaseRepository
from app.core.exceptions import ServiceException
import logging

ModelType = TypeVar("ModelType")
RepositoryType = TypeVar("RepositoryType", bound=BaseRepository)

class BaseController(Generic[ModelType, RepositoryType]):
    def __init__(self, repository: RepositoryType):
        self.repository = repository
        self.logger = logging.getLogger(self.__class__.__name__) 
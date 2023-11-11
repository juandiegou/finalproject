"""_summary_
    A customized exception Type
"""
from fastapi import HTTPException
class UnAuthorizedException(HTTPException):
    """_summary_
    This is an Custom exception of Unauthorized Exception type
    Args:
        HTTPException (_type_): A type of HTTPException
    """
    def __init_subclass__(cls) -> None:
        return super().__init_subclass__()
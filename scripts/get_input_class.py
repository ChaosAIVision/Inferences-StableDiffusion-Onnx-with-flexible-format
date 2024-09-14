import inspect
from dataclasses import dataclass, field, is_dataclass
from typing import List, Dict, Any, Tuple, Union, Optional, get_origin, get_args

@dataclass
class ParameterInfo:
    annotation: Optional[Any] = None
    default: Optional[Any] = None

    
@dataclass
class MethodInfo:
    parameters: Dict[str, ParameterInfo] = field(default_factory=dict)
    return_annotation: List[Optional[Any]] = field(default_factory=list)
    
    def get_raw_info(self) -> Dict[str, List[any]]:
        return {
            "parameters": list(self.parameters.keys()),
            "returns": [str(k) for k in self.return_annotation]
        }

@dataclass
class PyClassExtractor:
    class_object: Any
    class_name: str = field(init=False)
    methods_info: Dict[str, MethodInfo] = field(default_factory=dict, init=False)
    
    def __post_init__(self: object) -> None:
        self.class_name = self.class_object.__name__
        self._prepare_info()
        
    def __repr__(self: object) -> str:
        return f'PyClassExtractor: [{self.class_name}]'
    
    def _prepare_info(self: object) -> None:
        cls = self.class_object
        for method_name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            signature = inspect.signature(method)
            params_info = {}
            for param_name, param in signature.parameters.items():
                if param_name == "self":
                    continue
                annotation = param.annotation if param.annotation is not inspect.Parameter.empty else None
                default_param = param.default if param.default is not inspect.Parameter.empty else None
                params_info[param_name] = ParameterInfo(annotation=annotation, default=default_param)
            return_annotation = signature.return_annotation if signature.return_annotation is not inspect.Signature.empty else None
            if return_annotation is not None:
                origin = get_origin(return_annotation)
                if origin is Union:
                    return_annotation = list(get_args(return_annotation))
                else:
                    return_annotation = [return_annotation]
            if method_name == "forward" and len(return_annotation) > 1:
                if is_dataclass(return_annotation[-1]):
                    signature = inspect.signature(return_annotation[-1].__init__)
                    param_names = [k for k in signature.parameters.keys() if k != "self"]
                    self.methods_info[method_name] = MethodInfo(
                        parameters=params_info,
                        return_annotation=param_names
                    )
                    continue             
            self.methods_info[method_name] = MethodInfo(
                parameters=params_info,
                return_annotation=return_annotation
            )
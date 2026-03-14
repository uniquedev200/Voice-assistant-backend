import os
import importlib.util
import inspect
from typing import Dict, List, Any, Callable, Optional
import logging

logger = logging.getLogger(__name__)

_plugins: Dict[str, Dict[str, Any]] = {}


def discover_plugins() -> None:
    plugins_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins")
    
    if not os.path.exists(plugins_dir):
        logger.warning(f"Plugins directory not found: {plugins_dir}")
        return

    for filename in os.listdir(plugins_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            filepath = os.path.join(plugins_dir, filename)
            _load_plugin(filepath)


def _load_plugin(filepath: str) -> None:
    module_name = f"plugins.{os.path.splitext(os.path.basename(filepath))[0]}"
    
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec is None or spec.loader is None:
        return

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if hasattr(module, "register"):
        register_func = getattr(module, "register")
        plugin_info = register_func()
        _plugins[plugin_info["name"]] = plugin_info
        logger.info(f"Loaded plugin: {plugin_info['name']}")


def get_all_tools() -> List[Dict[str, Any]]:
    tools = []
    for plugin in _plugins.values():
        if "tools" in plugin:
            tools.extend(plugin["tools"])
    return tools


def route(tool_name: str, args: Dict[str, Any], context: Any, session: Any) -> Any:
    for plugin in _plugins.values():
        handler = plugin.get("handler")
        if handler:
            try:
                return handler(tool_name, args, context, session)
            except Exception as e:
                logger.error(f"Error in plugin handler: {e}")
                return {"error": str(e)}
    return {"error": f"Unknown tool: {tool_name}"}


def get_plugins() -> Dict[str, Dict[str, Any]]:
    return _plugins
import json
from typing import List, Any, Dict


class MessageObject:
    """A simple object to hold message data with attribute access"""
    def __init__(self, data_dict: Dict[str, Any]):
        self.__dict__.update(data_dict)


class ContentObject:
    """A simple object to hold content data with attribute access"""
    def __init__(self, data_dict: Dict[str, Any]):
        if isinstance(data_dict, dict):
            self.__dict__.update(data_dict)
        else:
            self.text = str(data_dict)


def serialize_message(msg) -> Dict[str, Any]:
    """Convert a Claude SDK message object to a serializable dictionary"""
    msg_dict = {
        "type": msg.__class__.__name__,
        "data": {}
    }
    
    # Serialize common attributes
    if hasattr(msg, "content"):
        msg_dict["data"]["content"] = serialize_content(msg.content)
    
    if hasattr(msg, "role"):
        msg_dict["data"]["role"] = msg.role
    
    if hasattr(msg, "model"):
        msg_dict["data"]["model"] = msg.model
    
    if hasattr(msg, "parent_tool_use_id"):
        msg_dict["data"]["parent_tool_use_id"] = msg.parent_tool_use_id
    
    if hasattr(msg, "result"):
        msg_dict["data"]["result"] = msg.result
    
    if hasattr(msg, "num_turns"):
        msg_dict["data"]["num_turns"] = msg.num_turns
    
    if hasattr(msg, "total_cost_usd"):
        msg_dict["data"]["total_cost_usd"] = msg.total_cost_usd
    
    if hasattr(msg, "duration_ms"):
        msg_dict["data"]["duration_ms"] = msg.duration_ms
    
    if hasattr(msg, "usage"):
        msg_dict["data"]["usage"] = msg.usage
    
    # Serialize any additional data attributes
    if hasattr(msg, "data") and isinstance(msg.data, dict):
        msg_dict["data"]["metadata"] = msg.data
    
    return msg_dict


def serialize_content(content) -> Any:
    """Serialize message content (can be string, list, or dict)"""
    if isinstance(content, str):
        return content
    
    if isinstance(content, list):
        serialized = []
        for item in content:
            if isinstance(item, dict):
                serialized.append(item)
            elif hasattr(item, "__dict__"):
                # Convert object to dict, capturing all relevant attributes
                item_dict = {"type": item.__class__.__name__}
                
                # Core text attributes
                if hasattr(item, "text"):
                    item_dict["text"] = item.text
                if hasattr(item, "name"):
                    item_dict["name"] = item.name
                if hasattr(item, "input"):
                    item_dict["input"] = item.input
                
                # ID tracking attributes - CRITICAL for subagent tracing
                if hasattr(item, "id"):
                    item_dict["id"] = item.id
                if hasattr(item, "tool_use_id"):
                    item_dict["tool_use_id"] = item.tool_use_id
                if hasattr(item, "parent_tool_use_id"):
                    item_dict["parent_tool_use_id"] = item.parent_tool_use_id
                
                # Content for ToolResultBlock - handle nested structures
                if hasattr(item, "content"):
                    if isinstance(item.content, str):
                        item_dict["content"] = item.content
                    elif isinstance(item.content, list):
                        # Recursively serialize nested content lists
                        item_dict["content"] = []
                        for nested_item in item.content:
                            if isinstance(nested_item, dict):
                                item_dict["content"].append(nested_item)
                            elif isinstance(nested_item, str):
                                item_dict["content"].append(nested_item)
                            elif hasattr(nested_item, "__dict__"):
                                # Handle objects within content
                                nested_dict = {}
                                if hasattr(nested_item, "type"):
                                    nested_dict["type"] = nested_item.type
                                if hasattr(nested_item, "text"):
                                    nested_dict["text"] = nested_item.text
                                item_dict["content"].append(nested_dict)
                            else:
                                item_dict["content"].append(str(nested_item))
                    else:
                        item_dict["content"] = str(item.content)
                
                # Error information
                if hasattr(item, "is_error"):
                    item_dict["is_error"] = item.is_error
                
                serialized.append(item_dict)
            else:
                serialized.append(str(item))
        return serialized
    
    if isinstance(content, dict):
        return content
    
    return str(content)


def deserialize_message(msg_dict: Dict[str, Any]) -> MessageObject:
    """Convert a dictionary back to a message-like object"""
    data = msg_dict.get("data", {})
    
    # Convert content back to objects if it's a list
    if "content" in data and isinstance(data["content"], list):
        data["content"] = [ContentObject(item) for item in data["content"]]
    
    return MessageObject(data)


def save_messages(messages: List[Any], filepath: str) -> None:
    """Save messages to a JSON file"""
    serialized = [serialize_message(msg) for msg in messages]
    with open(filepath, "w") as f:
        json.dump(serialized, f, indent=2)


def load_messages(filepath: str) -> List[MessageObject]:
    """Load messages from a JSON file as objects with attribute access"""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [deserialize_message(msg) for msg in data]


def load_messages_from_json(filepath: str) -> List[MessageObject]:
    """Load messages from a JSON file as objects with attribute access"""
    with open(filepath, "r") as f:
        data = json.load(f)
    return [deserialize_message(msg) for msg in data]

def greeting(name, prefix="Hello"):
    name = name.strip()
    if not name:
        raise ValueError("name must not be blank")
    return f"{prefix}, {name}!"

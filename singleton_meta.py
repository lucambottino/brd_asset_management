import threading
from weakref import WeakKeyDictionary

class SingletonMeta(type):
    """
    Singleton Metaclass: Ensures a class has only one instance and provides
    global access to that instance. This implementation uses per-class locks
    for thread safety and a weak reference dictionary to avoid memory leaks.
    """
    _instances = WeakKeyDictionary()

    def __init__(cls, name, bases, namespace):
        super().__init__(name, bases, namespace)
        # Assign a separate lock to each class using this metaclass
        cls._lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        # First check without the lock to optimize performance
        if cls not in SingletonMeta._instances:
            with cls._lock:
                # Double-check within the lock to ensure thread safety
                if cls not in SingletonMeta._instances:
                    instance = super().__call__(*args, **kwargs)
                    SingletonMeta._instances[cls] = instance
        return SingletonMeta._instances[cls]
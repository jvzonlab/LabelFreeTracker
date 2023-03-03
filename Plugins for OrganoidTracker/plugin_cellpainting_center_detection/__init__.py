from typing import Dict, Callable, Any, List

from organoid_tracker.gui.window import Window


def get_commands() -> Dict[str, Callable[[List[str]], int]]:
    # Command-line commands, used in python organoid_tracker.py <command>
    return {
        "train_nucleus_centers_from_transmitted_light": _train,
        "predict_nucleus_centers_from_transmitted_light": _predict
    }


def get_menu_items(window: Window) -> Dict[str, Any]:
    # Menu options for the OrganoidTracker GUI
    return {
        "Tools//Cellpainting-From transmitted light//Centers-Train nucleus center prediction...": lambda: _create_training_script(window),
        "Tools//Cellpainting-From transmitted light//Centers-Predict nucleus center positions...": lambda: _create_prediction_script(window)
    }


def _train(args: List[str]) -> int:
    from . import _training_script
    return _training_script.train(args)


def _predict(args: List[str]) -> int:
    from . import _prediction_script
    return _prediction_script.predict(args)


def _create_training_script(window: Window):
    from . import _script_creator
    _script_creator.create_training_script(window)


def _create_prediction_script(window: Window):
    from . import _script_creator
    _script_creator.create_prediction_script(window)

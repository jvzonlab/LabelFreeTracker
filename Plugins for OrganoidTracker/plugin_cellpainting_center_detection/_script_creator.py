import json
import os

from organoid_tracker.config import ConfigFile
from organoid_tracker.core import UserError
from organoid_tracker.gui import dialog, action
from organoid_tracker.gui.dialog import DefaultOption
from organoid_tracker.gui.window import Window
from organoid_tracker.imaging.list_io import FILES_LIST_EXTENSION
from organoid_tracker.util import run_script_creator


def create_training_script(window: Window):
    """Creates a folder and all scripts necessary to train a cell painting network."""
    # Save all pending changes
    files_json = action.to_experiment_list_file_structure(window.get_gui_experiment().get_active_tabs())
    if files_json is None:
        return  # Cancelled
    if len(files_json) == 0:
        raise UserError("No experiments loaded", "You haven't loaded any experiments, so we cannot train.")
    if len(files_json) == 1:
        if not dialog.popup_message_cancellable("Only one experiment selected", "You have only one experiment selected"
                                                " for training. It is recommended to train on multiple experiments at"
                                                " once, to prevent overfitting. Are you sure you want to continue?"):
            return

    # Locate the output folder
    output_folder = dialog.prompt_save_file("Directory for training", [("Folder", "*")])
    if output_folder is None:
        return  # Cancelled
    os.makedirs(output_folder, exist_ok=True)

    # Create all files
    with open(os.path.join(output_folder, "training_and_validation_dataset" + FILES_LIST_EXTENSION), "w") as handle:
        json.dump(files_json, handle)
    run_script_creator.create_run_script(output_folder, "train_nucleus_centers_from_transmitted_light")
    config = ConfigFile("train_nucleus_centers_from_transmitted_light", folder_name=output_folder)
    config.get_or_default("input_file", "training_and_validation_dataset" + FILES_LIST_EXTENSION)
    config.save()

    _show_success_confirmation(output_folder, "train_nucleus_centers_from_transmitted_light")


def create_prediction_script(window: Window):
    """Creates a folder and all scripts necessary to predict cell painting."""
    # Save all pending changes
    files_json = action.to_experiment_list_file_structure(window.get_gui_experiment().get_active_tabs())
    if files_json is None:
        return  # Cancelled
    if len(files_json) == 0:
        raise UserError("No experiments loaded", "You haven't loaded any experiments, so we cannot train.")

    if not dialog.popup_message_cancellable("Cell painting", "Please make sure you are currently viewing the transmitted"
                                                             " light channel.\n\nIf yes, then the next step will be to"
                                                             " select a folder with a trained cell painting network."
                                                             " This folder is likely named \"saved_model\" and has a"
                                                             " settings.json file in it."):
        return

    model_folder = dialog.prompt_directory("Please select a saved_model folder")
    if model_folder is None:
        return
    _validate_model_folder(model_folder)
    if not dialog.popup_message_cancellable("Cell painting", "Looks good!\n\nThe next step will be to select an output"
                                                             " folder."):
        return

    # Locate the output folder
    output_folder = dialog.prompt_save_file("Directory for prediction", [("Folder", "*")])
    if output_folder is None:
        return  # Cancelled
    os.makedirs(output_folder, exist_ok=True)

    try:
        transmitted_light_channel = window.display_settings.image_channel.index_one
    except ValueError:
        transmitted_light_channel = 1

    # Create all files
    with open(os.path.join(output_folder, "prediction_dataset" + FILES_LIST_EXTENSION), "w") as handle:
        json.dump(files_json, handle)
    run_script_creator.create_run_script(output_folder, "predict_nucleus_centers_from_transmitted_light")
    config = ConfigFile("predict_nucleus_centers_from_transmitted_light", folder_name=output_folder)
    config.get_or_default("input_file", "prediction_dataset" + FILES_LIST_EXTENSION)
    config.get_or_default("transmitted_light_channel", str(transmitted_light_channel))
    config.get_or_default("model_path", model_folder)
    config.get_or_default("output_folder", "output_images_{i}_{name}")
    config.save()

    _show_success_confirmation(output_folder, "predict_nucleus_centers_from_transmitted_light")


def _show_success_confirmation(output_folder: str, script_name: str):
    """Shows a popup that the config firles are created, and offers to open that directory."""
    if dialog.prompt_options("Configuration files created", f"The configuration files were created successfully. Please"
                                                            f" run the {script_name} script from that directory:\n\n{output_folder}",
                             option_1="Open that directory", option_default=DefaultOption.OK) == 1:
        dialog.open_file(output_folder)


def _validate_model_folder(model_folder: str):
    """Raises UserError if the user didn't select a valid model folder."""
    settings_file = os.path.join(model_folder, "settings.json")
    if not os.path.exists(settings_file):
        raise UserError("Not a model folder", "The selected folder is not a model folder, as it is missing a"
                                              " settings.json file.")
    with open(settings_file) as handle:
        settings = json.load(handle)
    if settings["type"] != "cell_painting":
        raise UserError("Wrong kind of model", f"Expected a cell painting model, but found a {settings['type']} model.")

"""
Talk2EP Demo — LangChain tool definitions.

All EnergyPlus tools route through the MCP client (stdio transport) to
the EnergyPlus-MCP server.  The SLIDERS knowledge tool calls the RAG
layer directly.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from mcp_client import mcp_client

# ---------------------------------------------------------------------------
# Lazy singleton for the SLIDERS knowledge layer
# ---------------------------------------------------------------------------
_knowledge = None


def _get_knowledge():
    global _knowledge
    if _knowledge is None:
        from knowledge import EnergyPlusKnowledge
        _knowledge = EnergyPlusKnowledge()
    return _knowledge


# ═══════════════════════════════════════════════════════════════════════════
# Knowledge tool  (calls SLIDERS directly — not an MCP tool)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def query_energyplus_knowledge(question: str) -> str:
    """Query the EnergyPlus Engineering Reference for building-physics
    knowledge, ASHRAE standards, default parameters, sizing factors, and
    simulation best-practices.  Use this when you need technical information
    to make informed modelling decisions BEFORE modifying the IDF file."""
    kb = _get_knowledge()
    return await kb.query(question)


# ═══════════════════════════════════════════════════════════════════════════
# Model loading & inspection  (via MCP)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def load_idf_model(idf_path: str) -> str:
    """Load an EnergyPlus IDF model file and return basic info (zone count,
    surface count, material count).  Always call this first before inspecting
    or modifying a model."""
    return await mcp_client.call_tool("load_idf_model", {"idf_path": idf_path})


@tool
async def get_model_summary(idf_path: str) -> str:
    """Get a high-level summary of the building model including Building name,
    Site:Location, SimulationControl settings, and EnergyPlus Version."""
    return await mcp_client.call_tool("get_model_summary", {"idf_path": idf_path})


@tool
async def list_available_files(include_example_files: bool = False,
                               include_weather_data: bool = False) -> str:
    """List IDF model files and weather files available on this machine
    (sample_files directory and optionally EnergyPlus example/weather dirs)."""
    return await mcp_client.call_tool("list_available_files", {
        "include_example_files": include_example_files,
        "include_weather_data": include_weather_data,
    })


@tool
async def validate_idf(idf_path: str) -> str:
    """Validate an IDF file for syntax errors, missing required objects,
    and broken material/construction references.  Run this AFTER making
    modifications to ensure the model is still valid."""
    return await mcp_client.call_tool("validate_idf", {"idf_path": idf_path})


@tool
async def check_simulation_settings(idf_path: str) -> str:
    """Inspect the current SimulationControl and RunPeriod settings of the
    model and list which fields can be modified."""
    return await mcp_client.call_tool("check_simulation_settings", {"idf_path": idf_path})


# ═══════════════════════════════════════════════════════════════════════════
# Detailed inspection tools  (via MCP)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def list_zones(idf_path: str) -> str:
    """List all thermal zones in the model with their Name, Origin
    coordinates, Type, Multiplier, Ceiling Height, and Volume."""
    return await mcp_client.call_tool("list_zones", {"idf_path": idf_path})


@tool
async def get_materials(idf_path: str) -> str:
    """Get all Material and Material:NoMass objects with their thermal
    properties (conductivity, density, specific heat, roughness, etc.)."""
    return await mcp_client.call_tool("get_materials", {"idf_path": idf_path})


@tool
async def inspect_people(idf_path: str) -> str:
    """Inspect all People objects — zone assignment, occupancy schedule,
    calculation method, density, and activity level."""
    return await mcp_client.call_tool("inspect_people", {"idf_path": idf_path})


@tool
async def inspect_lights(idf_path: str) -> str:
    """Inspect all Lights objects — zone, schedule, calculation method,
    design level, return air fraction, etc."""
    return await mcp_client.call_tool("inspect_lights", {"idf_path": idf_path})


@tool
async def inspect_electric_equipment(idf_path: str) -> str:
    """Inspect all ElectricEquipment objects — zone, schedule, design
    level, fraction radiant/latent/lost."""
    return await mcp_client.call_tool("inspect_electric_equipment", {"idf_path": idf_path})


# ═══════════════════════════════════════════════════════════════════════════
# Model modification tools  (via MCP)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def modify_people(idf_path: str,
                        modifications: List[Dict[str, Any]],
                        output_path: Optional[str] = None) -> str:
    """Modify People objects in the IDF model.

    Each item in `modifications` should have:
      - "target": "all" | "zone:<ZoneName>" | "name:<PeopleName>"
      - "field_updates": dict of field names → new values
        Supported fields: Number_of_People, People_per_Floor_Area,
        Fraction_Radiant, Activity_Level_Schedule_Name, etc.

    Returns the path of the saved (modified) IDF file.
    """
    args: dict[str, Any] = {"idf_path": idf_path, "modifications": modifications}
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("modify_people", args)


@tool
async def modify_lights(idf_path: str,
                        modifications: List[Dict[str, Any]],
                        output_path: Optional[str] = None) -> str:
    """Modify Lights objects in the IDF model.

    Each item in `modifications` should have:
      - "target": "all" | "zone:<ZoneName>" | "name:<LightsName>"
      - "field_updates": dict of field names → new values
        Supported fields: Lighting_Level, Watts_per_Floor_Area,
        Fraction_Radiant, Fraction_Visible, Return_Air_Fraction, etc.

    Returns the path of the saved (modified) IDF file.
    """
    args: dict[str, Any] = {"idf_path": idf_path, "modifications": modifications}
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("modify_lights", args)


@tool
async def modify_electric_equipment(idf_path: str,
                                    modifications: List[Dict[str, Any]],
                                    output_path: Optional[str] = None) -> str:
    """Modify ElectricEquipment objects in the IDF model.

    Each item in `modifications` should have:
      - "target": "all" | "zone:<ZoneName>" | "name:<EquipName>"
      - "field_updates": dict of field names → new values
        Supported fields: Design_Level, Watts_per_Floor_Area,
        Fraction_Radiant, Fraction_Latent, Fraction_Lost, etc.
    """
    args: dict[str, Any] = {"idf_path": idf_path, "modifications": modifications}
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("modify_electric_equipment", args)


@tool
async def modify_simulation_control(idf_path: str,
                                    field_updates: Dict[str, Any],
                                    output_path: Optional[str] = None) -> str:
    """Modify SimulationControl fields (e.g. Do_Zone_Sizing_Calculation,
    Do_System_Sizing_Calculation,
    Run_Simulation_for_Weather_File_Run_Periods, etc.)."""
    args: dict[str, Any] = {"idf_path": idf_path, "field_updates": field_updates}
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("modify_simulation_control", args)


@tool
async def modify_run_period(idf_path: str,
                            field_updates: Dict[str, Any],
                            run_period_index: int = 0,
                            output_path: Optional[str] = None) -> str:
    """Modify RunPeriod settings (Begin_Month, Begin_Day, End_Month,
    End_Day, Day_of_Week_for_Start_Day, etc.)."""
    args: dict[str, Any] = {
        "idf_path": idf_path,
        "field_updates": field_updates,
        "run_period_index": run_period_index,
    }
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("modify_run_period", args)


@tool
async def change_infiltration_by_mult(idf_path: str,
                                      mult: float,
                                      output_path: Optional[str] = None) -> str:
    """Scale ALL ZoneInfiltration:DesignFlowRate objects by a multiplier.
    E.g. mult=0.5 halves infiltration, mult=2.0 doubles it."""
    args: dict[str, Any] = {"idf_path": idf_path, "mult": mult}
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("change_infiltration_by_mult", args)


@tool
async def add_window_film_outside(idf_path: str,
                                  u_value: float = 4.94,
                                  shgc: float = 0.45,
                                  visible_transmittance: float = 0.66,
                                  output_path: Optional[str] = None) -> str:
    """Apply a window film (SimpleGlazingSystem) to ALL exterior windows.
    Specify U-value (W/m²·K), SHGC, and visible transmittance."""
    args: dict[str, Any] = {
        "idf_path": idf_path,
        "u_value": u_value,
        "shgc": shgc,
        "visible_transmittance": visible_transmittance,
    }
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("add_window_film_outside", args)


@tool
async def add_coating_outside(idf_path: str,
                              location: str = "wall",
                              solar_abs: float = 0.4,
                              thermal_abs: float = 0.9,
                              output_path: Optional[str] = None) -> str:
    """Apply a cool-roof or cool-wall coating by changing solar and thermal
    absorptance of exterior surface layers.  `location` must be "wall" or
    "roof"."""
    args: dict[str, Any] = {
        "idf_path": idf_path,
        "location": location,
        "solar_abs": solar_abs,
        "thermal_abs": thermal_abs,
    }
    if output_path is not None:
        args["output_path"] = output_path
    return await mcp_client.call_tool("add_coating_outside", args)


# ═══════════════════════════════════════════════════════════════════════════
# Simulation  (via MCP)
# ═══════════════════════════════════════════════════════════════════════════

@tool
async def run_energyplus_simulation(
    idf_path: str,
    weather_file: Optional[str] = None,
    output_directory: Optional[str] = None,
    annual: bool = True,
    design_day: bool = False,
    readvars: bool = True,
    expandobjects: bool = True,
) -> str:
    """Run the EnergyPlus simulation engine on an IDF file.

    Args:
        idf_path: Path to the IDF model.
        weather_file: Path, filename, or city name for the .epw weather file.
                      Supports fuzzy matching (e.g. "San Francisco").
        output_directory: Where to write outputs (auto-generated if None).
        annual: Run a full annual simulation (default True).
        design_day: Run design-day only (default False).
        readvars: Post-process with ReadVarsESO (default True).
        expandobjects: Pre-process with ExpandObjects (default True).

    Returns text with simulation duration, output files, and status.
    """
    args: dict[str, Any] = {
        "idf_path": idf_path,
        "annual": annual,
        "design_day": design_day,
        "readvars": readvars,
        "expandobjects": expandobjects,
    }
    if weather_file is not None:
        args["weather_file"] = weather_file
    if output_directory is not None:
        args["output_directory"] = output_directory
    return await mcp_client.call_tool("run_energyplus_simulation", args)


# ═══════════════════════════════════════════════════════════════════════════
# Collect all tools for the agent
# ═══════════════════════════════════════════════════════════════════════════

ALL_TOOLS = [
    # knowledge (direct SLIDERS call)
    query_energyplus_knowledge,
    # load / inspect (MCP)
    load_idf_model,
    get_model_summary,
    list_available_files,
    validate_idf,
    check_simulation_settings,
    list_zones,
    get_materials,
    inspect_people,
    inspect_lights,
    inspect_electric_equipment,
    # modify (MCP)
    modify_people,
    modify_lights,
    modify_electric_equipment,
    modify_simulation_control,
    modify_run_period,
    change_infiltration_by_mult,
    add_window_film_outside,
    add_coating_outside,
    # simulate (MCP)
    run_energyplus_simulation,
]

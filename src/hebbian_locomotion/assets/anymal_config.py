# src/hebbian_locomotion/assets/anymal_config.py

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR # Note: Variable name changed in recent versions

##
# Configuration - Actuators.
##

# We use this simple DC Motor model because it's stable and doesn't require external .pt files
ANYDRIVE_3_SIMPLE_ACTUATOR_CFG = DCMotorCfg(
    joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
    saturation_effort=120.0,
    effort_limit=80.0,
    velocity_limit=7.5,
    stiffness={".*": 40.0},
    damping={".*": 5.0},
)

##
# Configuration - Articulation.
##

ANYMAL_C_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # Points to the official NVIDIA asset
        usd_path=f"{ISAAC_NUCLEUS_DIR}/Robots/ANYbotics/ANYmal-C/anymal_c.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,     # Hip Adduction/Abduction
            ".*F_HFE": 0.4,   # Front Hip Flexion
            ".*H_HFE": -0.4,  # Hind Hip Flexion
            ".*F_KFE": -0.8,  # Front Knee
            ".*H_KFE": 0.8,   # Hind Knee
        },
    ),
    # CHANGED: Using the Simple DC Motor instead of LSTM
    actuators={"legs": ANYDRIVE_3_SIMPLE_ACTUATOR_CFG},
    soft_joint_pos_limit_factor=0.95,
)
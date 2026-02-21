#pragma once

// std lib
#include <stdlib.h>
#include <cmath>
#include <iostream>

// yaml cpp
#include <yaml-cpp/yaml.h>

// flightlib
#include "flightlib/bridges/unity_bridge.hpp"
#include "flightlib/common/command.hpp"
#include "flightlib/common/logger.hpp"
#include "flightlib/common/quad_state.hpp"
#include "flightlib/common/types.hpp"
#include "flightlib/envs/env_base.hpp"
#include "flightlib/objects/quadrotor.hpp"

namespace flightlib {

namespace quadenv {

enum Ctl : int {
  // observations: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
  kObs = 0,
  //
  kPos = 0,
  kNPos = 3,
  kOri = 3,
  kNOri = 4,
  kLinVel = 7,
  kNLinVel = 3,
  kAngVel = 10,
  kNAngVel = 3,
  kNObs = 13,
  // control actions
  kAct = 0,
  kNAct = 4,
};
};
class QuadrotorEnv final : public EnvBase {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  QuadrotorEnv();
  QuadrotorEnv(const std::string &cfg_path);
  ~QuadrotorEnv();

  // - public OpenAI-gym-style functions
  bool reset(Ref<Vector<>> obs, const bool random = true) override;
  Scalar step(const Ref<Vector<>> act, Ref<Vector<>> obs) override;

  // - public set functions
  bool loadParam(const YAML::Node &cfg);
  bool setMass(Scalar mass);
  bool setMotorTauInv(Scalar tau_inv);
  void setMotorInitMode(int mode);
  void setGoalPosition(Scalar x, Scalar y, Scalar z);

  // - public get functions
  bool getObs(Ref<Vector<>> obs) override;
  bool getAct(Ref<Vector<>> act) const;
  bool getAct(Command *const cmd) const;

  // - auxiliar functions
  bool isTerminalState(Scalar &reward) override;
  void addObjectsToUnity(std::shared_ptr<UnityBridge> bridge);

  friend std::ostream &operator<<(std::ostream &os,
                                  const QuadrotorEnv &quad_env);

 private:
  // quadrotor
  std::shared_ptr<Quadrotor> quadrotor_ptr_;
  QuadState quad_state_;
  Command cmd_;
  Logger logger_{"QaudrotorEnv"};

  // Define reward for training
  Scalar pos_coeff_, ori_coeff_, lin_vel_coeff_, ang_vel_coeff_, act_coeff_;

  // observations and actions (for RL)
  Vector<quadenv::kNObs> quad_obs_;
  Vector<quadenv::kNAct> quad_act_;

  // reward function design (for model-free reinforcement learning)
  Vector<quadenv::kNObs> goal_state_;

  // action and observation normalization (for learning)
  Vector<quadenv::kNAct> act_mean_;
  Vector<quadenv::kNAct> act_std_;
  Vector<quadenv::kNObs> obs_mean_ = Vector<quadenv::kNObs>::Zero();
  Vector<quadenv::kNObs> obs_std_ = Vector<quadenv::kNObs>::Ones();

  Vector<3> goal_pos_{0.0, 0.0, 5.0};
  int motor_init_mode_{0};  // 0=zero, 1=hover

  YAML::Node cfg_;
  Matrix<3, 2> world_box_;
};

}  // namespace flightlib
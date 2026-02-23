#include "flightlib/envs/quadrotor_env/quadrotor_env.hpp"

namespace flightlib {

QuadrotorEnv::QuadrotorEnv()
  : QuadrotorEnv(getenv("FLIGHTMARE_PATH") +
                 std::string("/flightlib/configs/quadrotor_env.yaml")) {}

QuadrotorEnv::QuadrotorEnv(const std::string &cfg_path)
  : EnvBase(),
    pos_coeff_(0.0),
    ori_coeff_(0.0),
    lin_vel_coeff_(0.0),
    ang_vel_coeff_(0.0),
    act_coeff_(0.0),
    goal_state_((Vector<quadenv::kNObs>() << 0.0, 0.0, 0.0,
                 1.0, 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0,
                 0.0, 0.0, 0.0)
                  .finished()) {
  // load configuration file
  YAML::Node cfg_ = YAML::LoadFile(cfg_path);

  quadrotor_ptr_ = std::make_shared<Quadrotor>();
  // update dynamics
  QuadrotorDynamics dynamics;
  dynamics.updateParams(cfg_);
  quadrotor_ptr_->updateDynamics(dynamics);

  // define a bounding box
  world_box_ << -20, 20, -20, 20, 0, 20;
  if (!quadrotor_ptr_->setWorldBox(world_box_)) {
    logger_.error("cannot set wolrd box");
  };

  // define input and output dimension for the environment
  obs_dim_ = quadenv::kNObs;
  act_dim_ = quadenv::kNAct;

  Scalar mass = quadrotor_ptr_->getMass();
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;

  // load parameters
  loadParam(cfg_);
}

QuadrotorEnv::~QuadrotorEnv() {}

bool QuadrotorEnv::reset(Ref<Vector<>> obs, const bool random) {
  quad_state_.setZero();
  quad_act_.setZero();

  if (random) {
    auto randu = [&](Scalar lo, Scalar hi) {
      return lo + (hi - lo) * (uniform_dist_(random_gen_) * 0.5 + 0.5);
    };

    // position: absolute world coordinates
    quad_state_.x(QS::POSX) = randu(spawn_pos_min_(0), spawn_pos_max_(0));
    quad_state_.x(QS::POSY) = randu(spawn_pos_min_(1), spawn_pos_max_(1));
    quad_state_.x(QS::POSZ) = randu(spawn_pos_min_(2), spawn_pos_max_(2));
    if (quad_state_.x(QS::POSZ) < 0.1)
      quad_state_.x(QS::POSZ) = 0.1;

    // linear velocity
    quad_state_.x(QS::VELX) = randu(spawn_vel_min_(0), spawn_vel_max_(0));
    quad_state_.x(QS::VELY) = randu(spawn_vel_min_(1), spawn_vel_max_(1));
    quad_state_.x(QS::VELZ) = randu(spawn_vel_min_(2), spawn_vel_max_(2));

    // angular velocity
    quad_state_.x(QS::OMEX) = randu(spawn_omega_min_(0), spawn_omega_max_(0));
    quad_state_.x(QS::OMEY) = randu(spawn_omega_min_(1), spawn_omega_max_(1));
    quad_state_.x(QS::OMEZ) = randu(spawn_omega_min_(2), spawn_omega_max_(2));

    // orientation: 0=upright, 1=full random
    if (spawn_ori_scale_ > 0.0) {
      quad_state_.x(QS::ATTW) = 1.0 + uniform_dist_(random_gen_) * spawn_ori_scale_;
      quad_state_.x(QS::ATTX) = uniform_dist_(random_gen_) * spawn_ori_scale_;
      quad_state_.x(QS::ATTY) = uniform_dist_(random_gen_) * spawn_ori_scale_;
      quad_state_.x(QS::ATTZ) = uniform_dist_(random_gen_) * spawn_ori_scale_;
      quad_state_.qx /= quad_state_.qx.norm();
    } else {
      quad_state_.x(QS::ATTW) = 1.0;
      quad_state_.x(QS::ATTX) = 0.0;
      quad_state_.x(QS::ATTY) = 0.0;
      quad_state_.x(QS::ATTZ) = 0.0;
    }
  }
  // reset quadrotor with random states
  quadrotor_ptr_->reset(quad_state_);

  // reset control command
  cmd_.t = 0.0;
  cmd_.thrusts.setZero();

  initHoverMotors();

  // obtain observations
  getObs(obs);
  return true;
}

bool QuadrotorEnv::getObs(Ref<Vector<>> obs) {
  quadrotor_ptr_->getState(&quad_state_);

  Quaternion q = quad_state_.q();
  if (q.w() < 0.0) {
    q.coeffs() = -q.coeffs();
  }

  Vector<3> pos_error = goal_pos_ - quad_state_.p;

  quad_obs_ << pos_error, q.w(), q.x(), q.y(), q.z(),
               quad_state_.v, quad_state_.w;

  obs.segment<quadenv::kNObs>(quadenv::kObs) = quad_obs_;
  return true;
}

Scalar QuadrotorEnv::step(const Ref<Vector<>> act, Ref<Vector<>> obs) {
  quad_act_ = act.cwiseProduct(act_std_) + act_mean_;
  cmd_.t += sim_dt_;
  cmd_.thrusts = quad_act_;

  // simulate quadrotor
  quadrotor_ptr_->run(cmd_, sim_dt_);

  // update observations
  getObs(obs);

  // ---------------------- reward function design
  // - position tracking
  Scalar pos_reward =
    pos_coeff_ * (quad_obs_.segment<quadenv::kNPos>(quadenv::kPos) -
                  goal_state_.segment<quadenv::kNPos>(quadenv::kPos))
                   .squaredNorm();
  // - orientation tracking
  Scalar ori_reward =
    ori_coeff_ * (quad_obs_.segment<quadenv::kNOri>(quadenv::kOri) -
                  goal_state_.segment<quadenv::kNOri>(quadenv::kOri))
                   .squaredNorm();
  // - linear velocity tracking
  Scalar lin_vel_reward =
    lin_vel_coeff_ * (quad_obs_.segment<quadenv::kNLinVel>(quadenv::kLinVel) -
                      goal_state_.segment<quadenv::kNLinVel>(quadenv::kLinVel))
                       .squaredNorm();
  // - angular velocity tracking
  Scalar ang_vel_reward =
    ang_vel_coeff_ * (quad_obs_.segment<quadenv::kNAngVel>(quadenv::kAngVel) -
                      goal_state_.segment<quadenv::kNAngVel>(quadenv::kAngVel))
                       .squaredNorm();

  // - control action penalty
  Scalar act_reward = act_coeff_ * act.cast<Scalar>().norm();

  Scalar total_reward =
    pos_reward + ori_reward + lin_vel_reward + ang_vel_reward + act_reward;

  // survival reward
  total_reward += 0.1;

  return total_reward;
}

bool QuadrotorEnv::isTerminalState(Scalar &reward) {
  if (quad_state_.x(QS::POSZ) <= 0.02) {
    reward = -0.02;
    return true;
  }
  reward = 0.0;
  return false;
}

bool QuadrotorEnv::loadParam(const YAML::Node &cfg) {
  if (cfg["quadrotor_env"]) {
    sim_dt_ = cfg["quadrotor_env"]["sim_dt"].as<Scalar>();
    max_t_ = cfg["quadrotor_env"]["max_t"].as<Scalar>();
  } else {
    return false;
  }

  if (cfg["rl"]) {
    // load reinforcement learning related parameters
    pos_coeff_ = cfg["rl"]["pos_coeff"].as<Scalar>();
    ori_coeff_ = cfg["rl"]["ori_coeff"].as<Scalar>();
    lin_vel_coeff_ = cfg["rl"]["lin_vel_coeff"].as<Scalar>();
    ang_vel_coeff_ = cfg["rl"]["ang_vel_coeff"].as<Scalar>();
    act_coeff_ = cfg["rl"]["act_coeff"].as<Scalar>();
  } else {
    return false;
  }
  return true;
}

void QuadrotorEnv::setMotorInitMode(int mode) {
  motor_init_mode_ = mode;
}

void QuadrotorEnv::initHoverMotors() {
  if (motor_init_mode_ != 1) return;
  Scalar hover_thrust = quadrotor_ptr_->getMass() * (-Gz) / 4.0;
  Vector<4> hover_thrusts = Vector<4>::Ones() * hover_thrust;
  QuadrotorDynamics dyn;
  quadrotor_ptr_->getDynamics(&dyn);
  Vector<4> hover_omega = dyn.motorThrustToOmega(hover_thrusts);
  quadrotor_ptr_->setMotorOmega(hover_omega);
  cmd_.thrusts = hover_thrusts;
}

void QuadrotorEnv::setGoalPosition(Scalar x, Scalar y, Scalar z) {
  goal_pos_ << x, y, z;
}

void QuadrotorEnv::setWorldBox(Ref<Vector<>> box) {
  // box: 6 floats [x_min, x_max, y_min, y_max, z_min, z_max]
  Matrix<3, 2> wb;
  wb << box(0), box(1), box(2), box(3), box(4), box(5);
  quadrotor_ptr_->setWorldBox(wb);
  world_box_ = wb;
}

void QuadrotorEnv::setSpawnRanges(Ref<Vector<>> ranges) {
  // ranges: 19 floats packed as:
  //   [pos_x_lo, pos_x_hi, pos_y_lo, pos_y_hi, pos_z_lo, pos_z_hi,
  //    vel_x_lo, vel_x_hi, vel_y_lo, vel_y_hi, vel_z_lo, vel_z_hi,
  //    omega_x_lo, omega_x_hi, omega_y_lo, omega_y_hi, omega_z_lo, omega_z_hi,
  //    ori_scale]
  spawn_pos_min_ << ranges(0), ranges(2), ranges(4);
  spawn_pos_max_ << ranges(1), ranges(3), ranges(5);
  spawn_vel_min_ << ranges(6), ranges(8), ranges(10);
  spawn_vel_max_ << ranges(7), ranges(9), ranges(11);
  spawn_omega_min_ << ranges(12), ranges(14), ranges(16);
  spawn_omega_max_ << ranges(13), ranges(15), ranges(17);
  spawn_ori_scale_ = ranges(18);
}

bool QuadrotorEnv::setMass(Scalar mass) {
  QuadrotorDynamics dyn;
  quadrotor_ptr_->getDynamics(&dyn);
  if (!dyn.setMass(mass)) return false;
  if (!quadrotor_ptr_->updateDynamics(dyn)) return false;
  act_mean_ = Vector<quadenv::kNAct>::Ones() * (-mass * Gz) / 4;
  act_std_ = Vector<quadenv::kNAct>::Ones() * (-mass * 2 * Gz) / 4;
  return true;
}

bool QuadrotorEnv::setMotorTauInv(Scalar tau_inv) {
  QuadrotorDynamics dyn;
  quadrotor_ptr_->getDynamics(&dyn);
  if (!dyn.setMotortauInv(tau_inv)) return false;
  return quadrotor_ptr_->updateDynamics(dyn);
}

bool QuadrotorEnv::getAct(Ref<Vector<>> act) const {
  if (cmd_.t >= 0.0 && quad_act_.allFinite()) {
    act = quad_act_;
    return true;
  }
  return false;
}

bool QuadrotorEnv::getAct(Command *const cmd) const {
  if (!cmd_.valid()) return false;
  *cmd = cmd_;
  return true;
}

void QuadrotorEnv::addObjectsToUnity(std::shared_ptr<UnityBridge> bridge) {
  bridge->addQuadrotor(quadrotor_ptr_);
}

std::ostream &operator<<(std::ostream &os, const QuadrotorEnv &quad_env) {
  os.precision(3);
  os << "Quadrotor Environment:\n"
     << "obs dim =            [" << quad_env.obs_dim_ << "]\n"
     << "act dim =            [" << quad_env.act_dim_ << "]\n"
     << "sim dt =             [" << quad_env.sim_dt_ << "]\n"
     << "max_t =              [" << quad_env.max_t_ << "]\n"
     << "act_mean =           [" << quad_env.act_mean_.transpose() << "]\n"
     << "act_std =            [" << quad_env.act_std_.transpose() << "]\n"
     << "obs_mean =           [" << quad_env.obs_mean_.transpose() << "]\n"
     << "obs_std =            [" << quad_env.obs_std_.transpose() << std::endl;
  os.precision();
  return os;
}

}  // namespace flightlib
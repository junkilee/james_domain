# vim: tabstop=2 softtabstop=2 shiftwidth=2 expandtab
from gym.envs.registration import registry, register, make, spec

register(
    id='JamesDomain',
    entry_point='james_domain.core:JamesEnv',
    kwargs={
      # 'name':'freeway'
    }
)

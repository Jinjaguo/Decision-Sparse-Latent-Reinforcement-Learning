# Next experiment from EXP17

EXP17 solved the formal recovery-candidate module: 92.86% safe availability, 41.67% decision demand, 82.86% oracle demand recovery, and 100% valid candidates. Drawer reached 85.71%, Bowl 100%, Stove 92.86%.

EXP18 now trains the action-consequence coordinator. Inputs are strictly pre-action: current state, route metadata and first proposed action. Whole demos are held out. The selector must beat the frozen default by at least 10 safe-success points, recover at least 60% of demand groups, capture at least 75% of oracle headroom, and not worsen safety.


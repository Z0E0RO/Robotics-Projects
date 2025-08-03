import numpy as np
import modern_robotics as mr
import pandas as pd


def IKinBodyIterates(Blist, M, T, thetalist0, eomg, ev):
    """Computes inverse kinematics in the body frame for an open chain robot

    :param Blist: The joint screw axes in the end-effector frame when the
                  manipulator is at the home position, in the format of a
                  matrix with axes as the columns
    :param M: The home configuration of the end-effector
    :param T: The desired end-effector configuration Tsd
    :param thetalist0: An initial guess of joint angles that are close to
                       satisfying Tsd
    :param eomg: A small positive tolerance on the end-effector orientation
                 error. The returned joint angles must give an end-effector
                 orientation error less than eomg
    :param ev: A small positive tolerance on the end-effector linear position
               error. The returned joint angles must give an end-effector
               position error less than ev
    :return thetalist: Joint angles that achieve T within the specified
                       tolerances,
    :return success: A logical value where TRUE means that the function found
                     a solution and FALSE means that it ran through the set
                     number of maximum iterations without finding a solution
                     within the tolerances eomg and ev.
    Uses an iterative Newton-Raphson root-finding method.
    The maximum number of iterations before the algorithm is terminated has
    been hardcoded in as a variable called maxiterations. It is set to 20 at
    the start of the function, but can be changed if needed.

    Example Input:
        Blist = np.array([[0, 0, -1, 2, 0,   0],
                          [0, 0,  0, 0, 1,   0],
                          [0, 0,  1, 0, 0, 0.1]]).T
        M = np.array([[-1, 0,  0, 0],
                      [ 0, 1,  0, 6],
                      [ 0, 0, -1, 2],
                      [ 0, 0,  0, 1]])
        T = np.array([[0, 1,  0,     -5],
                      [1, 0,  0,      4],
                      [0, 0, -1, 1.6858],
                      [0, 0,  0,      1]])
        thetalist0 = np.array([1.5, 2.5, 3])
        eomg = 0.01
        ev = 0.001
    Output:
        Iteration 0 :

        joint vector :
        [ 6.  -2.5  4.5 -5.   3.5  1.5]

        SE(3) end - effector config :
        [[-0.20506046  0.9761136   0.07178048 -0.43434202]
         [ 0.03383113  0.08036412 -0.99619128  0.15994292]
         [-0.97816443 -0.20185103 -0.04950253  0.07789615]
         [ 0.          0.          0.          1.        ]]

        error twist V_b :
        [ 0.07670657 -0.04201424  0.20515211 -0.01645098 -0.06999878  0.05694033]

        angular error magnitude ||omega_b|| : 0.22301677980707918

        linear error magnitude ||v_b|| : 0.09172058041018143
        (np.array([1.57073819, 2.999667, 3.14153913]), True)
    """
    thetalist = np.array(thetalist0).copy()
    i = 0
    maxiterations = 20
    end_effector_config = mr.FKinBody(M, Blist, thetalist)
    Vb = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(end_effector_config), T)))
    omega_b = np.linalg.norm([Vb[0:3]])
    v_b = np.linalg.norm([Vb[3:6]])
    err = omega_b > eomg \
          or v_b > ev
    joint_vector = []
    while err and i < maxiterations:
        print(f"Iteration {i} :")
        print("")
        print("joint vector :")
        print(thetalist)
        print("")
        print("SE(3) end - effector config :")
        print(end_effector_config)
        print("")
        print("error twist V_b :")
        print(Vb)
        print("")
        print(f"angular error magnitude ||omega_b|| : {omega_b}")
        print("")
        print(f"linear error magnitude ||v_b|| : {v_b}")
        joint_vector.append(thetalist.copy())
        i = i + 1
        thetalist = thetalist \
                    + np.dot(np.linalg.pinv(mr.JacobianBody(Blist, \
                                                         thetalist)), Vb)
        end_effector_config = mr.FKinBody(M, Blist, thetalist)
        Vb \
        = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(end_effector_config), T)))
        omega_b = np.linalg.norm([Vb[0:3]])
        v_b = np.linalg.norm([Vb[3:6]])
        err = omega_b > eomg \
              or v_b > ev
    print(f"Iteration {i} :")
    print("")
    print("joint vector :")
    print(thetalist)
    print("")
    print("SE(3) end - effector config :")
    print(end_effector_config)
    print("")
    print("error twist V_b :")
    print(Vb)
    print("")
    print(f"angular error magnitude ||omega_b|| : {omega_b}")
    print("")
    print(f"linear error magnitude ||v_b|| : {v_b}")
    joint_vector.append(thetalist.copy())
    
    pd.DataFrame(joint_vector).to_csv("iterates.csv", header = False, index = False)
    
    return (thetalist, not err)


T_desired = np.array([[0,1,0,-0.5],[0,0,-1,0.1],[-1,0,0,0.1],[0,0,0,1]])

M = np.array([[-1, 0, 0, 0.817],[0, 0, 1, 0.191], [0, 1, 0, -0.005], [0, 0, 0, 1]])

B_list = np.array([[0, 1, 0, 0.191, 0, 0.817], [0, 0, 1, 0.095, -0.817, 0], [0, 0, 1, 0.095, -0.392, 0],
                   [0, 0, 1, 0.095, 0, 0],[0, -1, 0, -0.082, 0, 0], [0, 0, 1, 0, 0, 0]]).T

e_omega = 0.001
e_vel = 0.0001

theta_list_0 = np.array([6.0000, -2.5000, 4.5000, -5.0000, 3.5000, 1.5000])

theta_list, converge = IKinBodyIterates(B_list, M, T_desired, theta_list_0, e_omega, e_vel)

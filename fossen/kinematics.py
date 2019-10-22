from scipy.spatial.transform import Rotation
from scipy.linalg import block_diag
import numpy as np


def Smtrx(lmda):
    if lmda.size > 3:
        S = np.zeros((lmda.shape[0], 3, 3))
    else:
        S = np.zeros((1, 3, 3))
        lmda = np.reshape(lmda, (1, 3))
    S[:, 0, 1] = -lmda[:, 2]
    S[:, 0, 2] = lmda[:, 1]
    S[:, 1, 0] = lmda[:, 2]
    S[:, 1, 2] = -lmda[:, 0]
    S[:, 2, 0] = -lmda[:, 1]
    S[:, 2, 1] = lmda[:, 0]
    return S


def Rzyx(phi, theta, psi):
    phi = np.float64(phi)
    theta = np.float64(theta)
    psi = np.float64(psi)
    euler = np.hstack(
        (np.reshape(phi, (phi.size, 1)), np.reshape(theta, (theta.size, 1)), np.reshape(psi, (psi.size, 1))))
    return Rotation.from_euler('xyz', euler, degrees=False).as_dcm()


def eulerang(phi, theta, psi, inverse=False):
    phi = np.float64(phi)
    theta = np.float64(theta)
    psi = np.float64(psi)
    assert not np.any(np.abs(theta) == np.pi / 2)
    J22 = np.zeros((phi.size, 3, 3))
    if inverse:
        J11 = np.transpose(Rzyx(phi, theta, psi), (0, 2, 1))
        J22[:, 0, 0] = 1
        J22[:, 0, 2] = -np.sin(theta)
        J22[:, 1, 1] = np.cos(phi)
        J22[:, 1, 2] = np.cos(theta) * np.sin(phi)
        J22[:, 2, 1] = -np.sin(phi)
        J22[:, 2, 2] = np.cos(theta) * np.cos(phi)
    else:
        J11 = Rzyx(phi, theta, psi)
        J22[:, 0, 0] = 1
        J22[:, 0, 1] = np.sin(phi) * np.tan(theta)
        J22[:, 0, 2] = np.cos(phi) * np.tan(theta)
        J22[:, 1, 1] = np.cos(phi)
        J22[:, 1, 2] = -np.sin(phi)
        J22[:, 2, 1] = np.sin(phi) / np.cos(theta)
        J22[:, 2, 2] = np.cos(phi) * np.cos(theta)
    J = np.block([[[J11, np.zeros((phi.size, 3, 3))],
                   [np.zeros((phi.size, 3, 3)), J22]]])
    return J, J11, J22


def Rquat(q):
    if q.ndim > 1:
        n = np.float64(q[:, 0])
        e1 = np.float64(q[:, 1])
        e2 = np.float64(q[:, 2])
        e3 = np.float64(q[:, 3])
    else:
        n = np.float64(q[0])
        e1 = np.float64(q[1])
        e2 = np.float64(q[2])
        e3 = np.float64(q[3])
    quat = np.hstack((np.reshape(e1, (e1.size, 1)), np.reshape(e2, (e2.size, 1)), np.reshape(e3, (e3.size, 1)),
                      np.reshape(n, (n.size, 1))))
    return Rotation.from_quat(quat, True).as_dcm()


def euler2q(phi, theta, psi):
    phi = np.float64(phi)
    theta = np.float64(theta)
    psi = np.float64(psi)
    euler = np.hstack(
        (np.reshape(phi, (phi.size, 1)), np.reshape(theta, (theta.size, 1)), np.reshape(psi, (psi.size, 1))))
    quat = Rotation.from_euler('xyz', euler, degrees=False).as_quat()
    quat[:, [0, 1, 2, 3]] = quat[:, [3, 0, 1, 2]]
    return quat


def quatern(q, inverse=False):
    if q.ndim > 1:
        n = q[:, 0]
        e1 = q[:, 1]
        e2 = q[:, 2]
        e3 = q[:, 3]
    else:
        n = np.float64(q[0])
        e1 = np.float64(q[1])
        e2 = np.float64(q[2])
        e3 = np.float64(q[3])
    if inverse:
        pass
    else:
        J11 = Rquat(q)
        J22 = np.zeros((n.size, 4, 3))
        J22[:, 0, 0] = -e1
        J22[:, 0, 1] = -e2
        J22[:, 0, 2] = -e3
        J22[:, 1, 0] = n
        J22[:, 1, 1] = -e3
        J22[:, 1, 2] = e2
        J22[:, 2, 0] = e3
        J22[:, 2, 1] = n
        J22[:, 2, 2] = -e1
        J22[:, 3, 0] = -e2
        J22[:, 3, 1] = e1
        J22[:, 3, 2] = n
        J22 *= 0.5
    J = np.block([[[J11, np.zeros((n.size, 3, 3))],
                   [np.zeros((n.size, 4, 3)), J22]]])
    return J, J11, J22


def q2euler(q):
    R = Rquat(q)
    assert np.all(np.abs(R[:, 2, 0]) <= 1)
    q = q[:,[1, 2, 3, 0]]
    euler = Rotation.from_quat(q).as_euler('xyz',False)
    return np.expand_dims(euler[:, 0], 0), np.expand_dims(euler[:, 1], 0), np.expand_dims(euler[:, 2], 0)

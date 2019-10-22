import numpy as np
from fossen import kinematics


def statelessdydt(model, input_scaler, output_scaler):
    """statelessdydt (t, y *args)
    Inputs
    t - time variable, not used for anything (placeholder for solve_ivp algorithm)
    y - ndarray of variables to be solved, y should be the following:
        y=[eta,Theta,etadot,Thetadot]
        eta = [N,E,D]
        Theta = [phi,theta,psi]
        etadot = [Ndot,Edot,Ddot]
        Thetadot = [phidot,thetadot,psidot]
    u - ndarray of control variables to be inputted, for nupiri muka this should be the following:
        u = [Tn,a0,a1,a2,a3,a4,a5]
    w - ndarray of environment variables to be inputted, currently not supported.
    Outputs
    dydt - the derivative of y evaluated  at t and y,
        ydot = [etadot, Thetadot, etaddot, Thetaddot]
        etaddot = [Nddot,Eddot,Dddot],
        Thetaddot = [phiddot, thetaddot, psiddot]
    """
    # TODO unit test this function
    # TODO Need to scale inputs
    # TODO Easier to use odeint, setup inputs for odeint
    def gstar(t, y, u, w=None):
        """The inputs should be like below:
        t - can be anything
        y - 2D array of KxN dimensions, where:
            N = number of variables that are being integrated (should be 12)
            K = number of samples (for vectorisation)
        u - 2D array of Ux1 dimensions, where:
            U = number of control variables over 1 control cycle (0.1 Hz)
        w - Not yet supported.
        """
        if not w is None:
            raise NotImplementedError
        # make sure all inputs have the right number of dimensions
        if y.ndim < 2:
            y = np.expand_dims(y, 0)
        if u.ndim < 2:
            u = np.expand_dims(u, 0)
        # Get the variables
        eta = y[:, 0:3]
        Theta = y[:, 3:6]
        phi = Theta[:, 0]
        theta = Theta[:, 1]
        psi = Theta[:, 2]
        etadot = y[:, 6:9]
        Thetadot = y[:, 9:]
        # convert etadot and Thetadot to nabla
        _, J11, J22 = kinematics.eulerang(phi, theta, psi, True)
        v = np.matmul(J11, np.expand_dims(etadot, 2)).squeeze(2)
        omega = np.matmul(J22, np.expand_dims(Thetadot, 2)).squeeze(2)
        nabla = np.hstack((v, omega))
        Z = eta[:, 2]
        if Z.ndim < 2:
            Z = np.expand_dims(Z, 1)
        # Get the input to the network
        if w is None:
            X = np.hstack((nabla, Z, Theta, u))
        else:
            raise NotImplementedError
        # scale the data down to inputs
        X = input_scaler.transform(X)
        # Predict the acceleration vector (while scaling the data to real)
        a = output_scaler.inverse_transform(model.model.predict(X))
        # Convert the acceleration vector into inertia frame
        J, _, _ = kinematics.eulerang(phi, theta, psi, False)
        etaddot = np.matmul(J, np.expand_dims(a, 2)).squeeze(2)
        # return y' = etaddot, etadot
        ydot = np.hstack((etadot, Thetadot, etaddot))
        return ydot

    return gstar

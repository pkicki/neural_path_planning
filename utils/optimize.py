import scipy.optimize as spo
import tensorflow as tf

from models.planner import DummyPlanner


def nelder_mead(f, x0, epochs=10, maxiter=15):
    for i in range(epochs):
        r = spo.minimize(f, x0, method='Nelder-Mead', options={'maxiter': maxiter})
        x0 = r.x
        if r.fun == 0.:
            break
    return r.x

def tf_grad_dummy(p, N, data, loss):
    optimizer = tf.keras.optimizers.Adam(1e-3)
    dp = DummyPlanner(p, N)
    for i in range(100):
       with tf.GradientTape(persistent=True) as tape:
           plan, cps = dp(data)
           model_loss, x_path, y_path, th_path, curvature = loss.curvature(plan)
       grads = tape.gradient(model_loss, dp.trainable_variables)
       optimizer.apply_gradients(zip(grads, dp.trainable_variables))
       print(model_loss)
       if tf.equal(model_loss, 0.):
           break
    return plan

import tensorflow as tf
a = tf.constant(2)
b = tf.constant(3)
x = tf.add(a,b)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
writer = tf.summary.FileWriter('./graphs',sess.graph)
print sess.run(x, feed_dict=None, options=None, run_metadata=None)

writer.close()

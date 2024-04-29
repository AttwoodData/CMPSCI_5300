def train_model(model, X, y, callback=None):
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall', 'accuracy', 'AUC'])
	return model.fit(X, y, epochs=100, callbacks=callback, verbose=0)

def print_debug():
	print('debug')


def mp_train(model, X, y, X_test, y_test, callback=None):
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['Recall', 'accuracy', 'AUC'])
	return model.fit(X, y, validation_data=(X_test, y_test), epochs=500, callbacks=callback, verbose=0)
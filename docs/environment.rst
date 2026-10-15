Environment Variables
---------------------

The BatBot API and CLI have environment variables (envars) that allow you to configure global settings
and configurations.

   - ``BATBOT_VERBOSE`` or the legacy ``VERBOSE`` (default: not set)
      A verbosity flag that can be set to turn on debug logging.  Defaults to "not set", which translates
      to no debug logging.  Setting this value to anything will turn on debug logging
      (e.g., ``VERBOSE=1``).
   - ``BATBOT_CLASSIFIER_CONFIG`` or the legacy ``CLASSIFIER_CONFIG`` (default: ``mobilenet``)
      Selects the classifier model configuration.
   - ``BATBOT_CLASSIFIER_BATCH_SIZE`` or the legacy ``CLASSIFIER_BATCH_SIZE`` (default: ``10``)
      Limits the number of 224-by-224 spectrogram windows sent to ONNX Runtime
      in one inference call.
   - ``BATBOT_REDUCED`` (default: not set)
      Set to ``1`` to limit each ONNX Runtime session to two intra-op and
      inter-op threads. This can reduce contention when using several workers.
   - ``BATBOT_CACHE_DIR`` (default: Pooch's platform-specific user cache)
      Overrides the model/example cache root. Native paths, environment variables, and
      the home-directory shorthand are supported. Containers default to
      ``/tmp/batbot-cache`` so non-root and arbitrary-UID runs can cache models.

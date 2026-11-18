'''Start the digit generation API.'''

import uvicorn

from digitgan.api import create_app
from digitgan.config import SERVICE_LOG_FORMAT, EnvironmentSettings, configure_logging, load_config


def main() -> None:
    '''Load the configuration and serve the API with uvicorn.'''
    settings = EnvironmentSettings()
    configure_logging(settings.log_level, SERVICE_LOG_FORMAT)
    config = load_config(settings=settings)
    # log_config=None keeps the logging set up above instead of uvicorn's own format
    uvicorn.run(
        create_app(config),
        host=config.api.host,
        port=config.api.port,
        log_config=None,
        access_log=False,
    )


if __name__ == '__main__':
    main()

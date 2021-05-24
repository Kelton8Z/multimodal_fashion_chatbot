import os
import webbrowser
from pathlib import Path

from jina import Flow
from jina.helper import countdown
from jina.logging import default_logger
from jina.parsers.helloworld import set_hw_parser

if __name__ == '__main__':
    from helper import (
        print_result,
        write_html,
        download_data,
        index_generator,
        query_generator,
        colored,
    )
    from executors import MyEncoder, MyIndexer, MyEvaluator
else:
    from .helper import (
        print_result,
        write_html,
        download_data,
        index_generator,
        query_generator,
        colored,
    )
    from .executors import MyEncoder, MyIndexer, MyEvaluator

cur_dir = os.path.dirname(os.path.abspath(__file__))


def hello_world(args):
    """
    Runs Jina's Hello World.

    Usage:
        Use it via CLI :command:`jina hello-world`.

    Description:
        It downloads Fashion-MNIST dataset and :term:`Indexer<indexes>` 50,000 images.
        The index is stored into 4 *shards*. It randomly samples 128 unseen images as :term:`Queries<Searching>`
        Results are shown in a webpage.

    More options can be found in :command:`jina hello-world --help`

    :param args: Argparse object
    """

    Path(args.workdir).mkdir(parents=True, exist_ok=True)

    targets = {
        'index-labels': {
            'url': args.index_labels_url,
            'filename': os.path.join(args.workdir, 'index-labels'),
        },
        'query-labels': {
            'url': args.query_labels_url,
            'filename': os.path.join(args.workdir, 'query-labels'),
        },
        'index': {
            'url': args.index_data_url,
            'filename': os.path.join(args.workdir, 'index-original'),
        },
        'query': {
            'url': args.query_data_url,
            'filename': os.path.join(args.workdir, 'query-original'),
        },
    }

    # download the data
    download_data(targets, args.download_proxy)

    # reduce the network load by using `fp16`, or even `uint8`
    os.environ['JINA_ARRAY_QUANT'] = 'fp16'
    os.environ['HW_WORKDIR'] = args.workdir

    # now comes the real work
    f = Flow().add(uses=MyEncoder, parallel=2).add(uses=MyIndexer).add(uses=MyEvaluator)
    # run it!
    with f:
        f.index(
            index_generator(num_docs=targets['index']['data'].shape[0], target=targets),
            request_size=args.request_size,
        )
        f.use_rest_gateway(args.port_expose)
        # wait for couple of seconds
        countdown(
            3,
            reason=colored(
                'behold! im going to switch to query mode',
                'cyan',
                attrs=['underline', 'bold', 'reverse'],
            ),
        )

        # f.search(
        #     query_generator(
        #         num_docs=args.num_query, target=targets, with_groundtruth=True
        #     ),
        #     shuffle=True,
        #     on_done=print_result,
        #     request_size=args.request_size,
        #     parameters={'top_k': args.top_k},
        # )
        '''
        f.post(
            '/eval',
            query_generator(
                num_docs=args.num_query, target=targets, with_groundtruth=True
            ),
            shuffle=True,
            on_done=print_result,
            request_size=args.request_size,
            parameters={'top_k': args.top_k},
        )'''

        # write result to html
        #write_html(os.path.join('static/chatbot.html'))
        url_html_path = 'file://' + os.path.abspath(
            os.path.join(
                os.path.dirname(os.path.realpath(__file__)), 'static/chatbot.html'
            )
        )
        try:
            webbrowser.open(url_html_path, new=2)
        except:
            pass  # intentional pass, browser support isn't cross-platform
        finally:
            default_logger.success(
                f'You should see a demo page opened in your browser, '
                f'if not, you may open {url_html_path} manually'
            )

        f.block()
        # if not args.unblock_query_flow:
        #     f.block()


if __name__ == '__main__':
    args = set_hw_parser().parse_args()
    hello_world(args)
